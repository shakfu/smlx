/*
 * libsmlx implementation. Public API in smlx.h.
 *
 * Assumed weight naming (HF / mlx-lm Llama):
 *   model.embed_tokens.weight                          [vocab, dim]
 *   model.layers.{i}.input_layernorm.weight            [dim]
 *   model.layers.{i}.self_attn.{q,k,v,o}_proj.weight   [out, in]
 *   model.layers.{i}.post_attention_layernorm.weight   [dim]
 *   model.layers.{i}.mlp.{gate,up,down}_proj.weight    [out, in]
 *   model.norm.weight                                  [dim]
 *   lm_head.weight                                     [vocab, dim]  (optional; tied otherwise)
 *
 * Internal types (Weights, QLinear, LayerW, KVCache) are not in the public
 * header -- they're owned by the opaque smlx_model / smlx_session handles.
 */

#include "smlx.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mlx/c/mlx.h"

/* ---------- error handling ---------- */

#define CHECK(call) do { \
  if ((call) != 0) { \
    fprintf(stderr, "mlx call failed at %s:%d: %s\n", __FILE__, __LINE__, #call); \
    exit(1); \
  } \
} while (0)

/* smlx_config struct is defined in smlx.h. */

int smlx_config_load(smlx_config* c, const char* path) {
  memset(c, 0, sizeof(*c));
  c->rope_theta = 10000.0f;
  c->norm_eps = 1e-5f;
  c->rope_factor = 1.0f;
  c->rope_low_freq_factor = 1.0f;
  c->rope_high_freq_factor = 4.0f;
  c->rope_original_max_pos = 0;
  FILE* f = fopen(path, "r");
  if (!f) { fprintf(stderr, "smlx: cannot open config %s\n", path); return -1; }
  char line[256];
  while (fgets(line, sizeof(line), f)) {
    char k[64]; char v[128];
    if (sscanf(line, " %63[^= \t] = %127s", k, v) != 2) continue;
    if      (!strcmp(k, "n_layers"))   c->n_layers   = atoi(v);
    else if (!strcmp(k, "dim"))        c->dim        = atoi(v);
    else if (!strcmp(k, "n_heads"))    c->n_heads    = atoi(v);
    else if (!strcmp(k, "n_kv_heads")) c->n_kv_heads = atoi(v);
    else if (!strcmp(k, "head_dim"))   c->head_dim   = atoi(v);
    else if (!strcmp(k, "hidden_dim")) c->hidden_dim = atoi(v);
    else if (!strcmp(k, "vocab_size")) c->vocab_size = atoi(v);
    else if (!strcmp(k, "rope_theta")) c->rope_theta = (float)atof(v);
    else if (!strcmp(k, "norm_eps"))   c->norm_eps   = (float)atof(v);
    else if (!strcmp(k, "rope_factor"))            c->rope_factor            = (float)atof(v);
    else if (!strcmp(k, "rope_low_freq_factor"))   c->rope_low_freq_factor   = (float)atof(v);
    else if (!strcmp(k, "rope_high_freq_factor"))  c->rope_high_freq_factor  = (float)atof(v);
    else if (!strcmp(k, "rope_original_max_pos"))  c->rope_original_max_pos  = atoi(v);
    else if (!strcmp(k, "q_bits"))                 c->q_bits                 = atoi(v);
    else if (!strcmp(k, "q_group_size"))           c->q_group_size           = atoi(v);
  }
  fclose(f);
  if (c->n_layers <= 0 || c->dim <= 0) {
    fprintf(stderr, "smlx: config missing required fields\n");
    return -1;
  }
  return 0;
}

/* ---------- weight handles ---------- */

/* A linear layer: dense (w pre-transposed, scales/biases empty) or quantized
 * (w = packed uint32 [out, in/pack], scales/biases populated). The forward
 * branches on smlx_config.quantized.
 *
 * Note: `biases` (plural) are quantization zero-points, used inside
 * quantized_matmul. `out_bias` (singular) is a projection bias added AFTER the
 * matmul (Qwen 2.x Q/K/V projections); empty handle when the layer has none. */
typedef struct {
  mlx_array w;
  mlx_array scales;
  mlx_array biases;
  mlx_array out_bias;
} QLinear;

typedef struct {
  mlx_array attn_norm;
  mlx_array ffn_norm;
  QLinear   wq, wk, wv, wo;
  QLinear   w_gate, w_up, w_down;
  /* Qwen3 QK-Norm: per-head RMSNorm weights [head_dim], applied to Q and K
   * before RoPE. Empty handles when the model has no QK-Norm. */
  mlx_array q_norm, k_norm;
} LayerW;

typedef struct {
  mlx_array tok_embed;          /* [vocab, dim], dense (dequantized if needed) */
  mlx_array norm;               /* [dim]        */
  QLinear   lm_head;            /* dense: w = embed.T; quantized: same triple as tied embed */
  LayerW* layers;               /* length n_layers */
  /* RoPE precomputed inverse frequencies (length head_dim/2). Empty handle = disabled. */
  mlx_array rope_freqs;
  float*    rope_freqs_buf;     /* owns backing store for rope_freqs */
  bool      use_rope_freqs;
} Weights;

/* Look up a weight by key and remove it from the map (we keep our own ref). */
static mlx_array take_weight(mlx_map_string_to_array data, const char* key, bool required) {
  mlx_array a = mlx_array_new();
  if (mlx_map_string_to_array_get(&a, data, key) != 0) {
    if (required) { fprintf(stderr, "missing weight: %s\n", key); exit(1); }
    mlx_array_free(a);
    return mlx_array_new();
  }
  return a;
}

/* Transpose a 2D weight to make it directly usable on the right of matmul. */
static mlx_array transpose_2d(mlx_array w, mlx_stream s) {
  mlx_array t = mlx_array_new();
  CHECK(mlx_transpose(&t, w, s));
  mlx_array_free(w);
  return t;
}

static void eval_weight(mlx_array a) { CHECK(mlx_array_eval(a)); }

/* Build Llama-3 scaled RoPE "freqs" for mlx_fast_rope.
 *
 * mlx_fast_rope's `freqs` parameter takes PERIODS, i.e. theta^(2i/d) -- not
 * angular inverse-frequencies. This matches mlx-lm's Llama3RoPE. The scaling
 * therefore works on periods:
 *   period_i  = theta^(2i / head_dim),    i in [0, head_dim/2)
 *   wavelen_i = 2*pi * period_i
 *   if wavelen < high_wl: keep
 *   if wavelen > low_wl : period * factor       (stretch long waves further)
 *   else                : period / ((1-s)/factor + s),  s = (old_ctx/wl - lo)/(hi-lo)
 */
static void build_rope_freqs(Weights* W, const smlx_config* c) {
  if (c->rope_original_max_pos <= 0 || c->rope_factor == 1.0f) {
    W->use_rope_freqs = false;
    W->rope_freqs = mlx_array_new();
    W->rope_freqs_buf = NULL;
    return;
  }
  int half = c->head_dim / 2;
  float* buf = malloc(sizeof(float) * (size_t)half);
  const float theta  = c->rope_theta;
  const float factor = c->rope_factor;
  const float lo_f   = c->rope_low_freq_factor;
  const float hi_f   = c->rope_high_freq_factor;
  const float old_ctx = (float)c->rope_original_max_pos;
  const float low_wl  = old_ctx / lo_f;
  const float high_wl = old_ctx / hi_f;
  for (int i = 0; i < half; i++) {
    float p = powf(theta, (float)(2 * i) / (float)c->head_dim);
    float wl = 2.0f * (float)M_PI * p;
    float np;
    if (wl < high_wl) {
      np = p;
    } else if (wl > low_wl) {
      np = p * factor;
    } else {
      float smooth = (old_ctx / wl - lo_f) / (hi_f - lo_f);
      np = p / ((1.0f - smooth) / factor + smooth);
    }
    buf[i] = np;
  }
  int sh[1] = { half };
  W->rope_freqs = mlx_array_new_data(buf, sh, 1, MLX_FLOAT32);
  W->rope_freqs_buf = buf;
  W->use_rope_freqs = true;
  CHECK(mlx_array_eval(W->rope_freqs));
}

/* Load a linear's weights. For dense: takes `.weight`, transposes to [in,out].
 * For quantized: takes `.weight` (packed uint32), `.scales`, `.biases` as-is.
 * Always tries to load an optional projection bias `.bias` (Qwen 2.x QKV). */
static void load_qlinear(QLinear* q, mlx_map_string_to_array data, const char* base,
                         const smlx_config* c, mlx_stream s) {
  char key[160];
  snprintf(key, sizeof(key), "%s.weight", base);
  if (c->quantized) {
    q->w = take_weight(data, key, true);
    snprintf(key, sizeof(key), "%s.scales", base); q->scales = take_weight(data, key, true);
    snprintf(key, sizeof(key), "%s.biases", base); q->biases = take_weight(data, key, true);
  } else {
    q->w = transpose_2d(take_weight(data, key, true), s);
    q->scales = mlx_array_new();
    q->biases = mlx_array_new();
  }
  snprintf(key, sizeof(key), "%s.bias", base);
  q->out_bias = take_weight(data, key, false);  /* empty handle if absent */
}

static void free_qlinear(QLinear* q) {
  mlx_array_free(q->w); mlx_array_free(q->scales);
  mlx_array_free(q->biases); mlx_array_free(q->out_bias);
}

static void eval_qlinear(QLinear* q) {
  eval_weight(q->w);
  if (q->scales.ctx) eval_weight(q->scales);
  if (q->biases.ctx) eval_weight(q->biases);
  if (q->out_bias.ctx) eval_weight(q->out_bias);
}

/* Apply: y = x @ W.T (+ out_bias) -- for both dense and quantized variants. */
static mlx_array qlinear_apply(const QLinear* q, mlx_array x, const smlx_config* c, mlx_stream s) {
  mlx_array r = mlx_array_new();
  if (c->quantized) {
    mlx_optional_int gs = { .value = c->q_group_size, .has_value = true };
    mlx_optional_int bb = { .value = c->q_bits,       .has_value = true };
    CHECK(mlx_quantized_matmul(&r, x, q->w, q->scales, q->biases,
                               /*transpose=*/true, gs, bb, "affine", s));
  } else {
    CHECK(mlx_matmul(&r, x, q->w, s));
  }
  mlx_array_free(x);
  if (q->out_bias.ctx) {
    mlx_array t = mlx_array_new();
    CHECK(mlx_add(&t, r, q->out_bias, s));  /* broadcasts [out] over [..., out] */
    mlx_array_free(r);
    r = t;
  }
  return r;
}

static void load_weights(Weights* W, smlx_config* c, const char* path, mlx_stream s) {
  mlx_map_string_to_array data = mlx_map_string_to_array_new();
  mlx_map_string_to_string meta = mlx_map_string_to_string_new();
  CHECK(mlx_load_safetensors(&data, &meta, path, s));

  /* Auto-detect quantization: scales for layer 0 q_proj => quantized model. */
  {
    mlx_array probe = mlx_array_new();
    int rc = mlx_map_string_to_array_get(&probe, data,
              "model.layers.0.self_attn.q_proj.scales");
    mlx_array_free(probe);
    if (rc == 0) {
      c->quantized = true;
      if (c->q_bits == 0)       c->q_bits = 4;
      if (c->q_group_size == 0) c->q_group_size = 64;
      fprintf(stderr, "[smlx] quantized model: bits=%d group_size=%d\n",
              c->q_bits, c->q_group_size);
    }
  }

  /* Auto-detect architecture variants on layer 0:
   *   q_proj.bias  => Qwen 2.x style QKV projection bias
   *   q_norm.weight => Qwen 3 style QK-Norm */
  {
    mlx_array probe = mlx_array_new();
    if (mlx_map_string_to_array_get(&probe, data,
          "model.layers.0.self_attn.q_proj.bias") == 0) {
      c->attn_qkv_bias = true;
    }
    mlx_array_free(probe);
    probe = mlx_array_new();
    if (mlx_map_string_to_array_get(&probe, data,
          "model.layers.0.self_attn.q_norm.weight") == 0) {
      c->qk_norm = true;
    }
    mlx_array_free(probe);
    if (c->attn_qkv_bias) fprintf(stderr, "[smlx] QKV projection bias (Qwen 2.x)\n");
    if (c->qk_norm)       fprintf(stderr, "[smlx] QK-Norm (Qwen 3)\n");
  }

  /* Embedding: dequantize to dense for fast take-based lookup. */
  if (c->quantized) {
    mlx_array ew = take_weight(data, "model.embed_tokens.weight", true);
    mlx_array es = take_weight(data, "model.embed_tokens.scales", true);
    mlx_array eb = take_weight(data, "model.embed_tokens.biases", true);
    mlx_array deq = mlx_array_new();
    mlx_optional_int gs = { .value = c->q_group_size, .has_value = true };
    mlx_optional_int bb = { .value = c->q_bits,       .has_value = true };
    mlx_optional_dtype dt = { .has_value = false };
    mlx_array null_arr = { 0 };
    CHECK(mlx_dequantize(&deq, ew, es, eb, gs, bb, "affine", null_arr, dt, s));
    mlx_array_free(ew); mlx_array_free(es); mlx_array_free(eb);
    W->tok_embed = deq;
  } else {
    W->tok_embed = take_weight(data, "model.embed_tokens.weight", true);
  }
  W->norm = take_weight(data, "model.norm.weight", true);

  /* LM head: prefer explicit lm_head if present, else tied embeddings. */
  {
    mlx_array probe = mlx_array_new();
    if (mlx_map_string_to_array_get(&probe, data, "lm_head.weight") == 0) {
      /* Untied. Re-use load_qlinear with synthetic base. */
      mlx_array_free(probe);
      load_qlinear(&W->lm_head, data, "lm_head", c, s);
    } else {
      mlx_array_free(probe);
      /* Tied. */
      if (c->quantized) {
        /* Re-load the quantized embedding triple as the lm_head linear; we still
         * have the dense `tok_embed` for embedding lookup. */
        W->lm_head.w      = take_weight(data, "model.embed_tokens.weight", false);
        W->lm_head.scales = take_weight(data, "model.embed_tokens.scales", false);
        W->lm_head.biases = take_weight(data, "model.embed_tokens.biases", false);
        /* If they were already taken (above) the map returns nothing; in that case
         * just re-dequantize. We avoided that by re-reading from the freshly
         * loaded map: `take_weight` errored only on `required=true`. Here, if
         * weight is missing we synthesize from dense tok_embed via transpose. */
        if (!W->lm_head.w.ctx) {
          fprintf(stderr, "[smlx] tied lm_head but no embed triple; bug?\n");
          exit(1);
        }
      } else {
        mlx_array tied = mlx_array_new();
        CHECK(mlx_transpose(&tied, W->tok_embed, s));
        W->lm_head.w = tied;
        W->lm_head.scales = mlx_array_new();
        W->lm_head.biases = mlx_array_new();
      }
    }
  }

  W->layers = calloc(c->n_layers, sizeof(LayerW));
  char base[160];
  for (int i = 0; i < c->n_layers; i++) {
    LayerW* L = &W->layers[i];
    char key[160];
    snprintf(key, sizeof(key), "model.layers.%d.input_layernorm.weight", i);
    L->attn_norm = take_weight(data, key, true);
    snprintf(key, sizeof(key), "model.layers.%d.post_attention_layernorm.weight", i);
    L->ffn_norm = take_weight(data, key, true);
    #define LOADQ(field, suffix) \
      snprintf(base, sizeof(base), "model.layers.%d." suffix, i); \
      load_qlinear(&L->field, data, base, c, s)
    LOADQ(wq,     "self_attn.q_proj");
    LOADQ(wk,     "self_attn.k_proj");
    LOADQ(wv,     "self_attn.v_proj");
    LOADQ(wo,     "self_attn.o_proj");
    LOADQ(w_gate, "mlp.gate_proj");
    LOADQ(w_up,   "mlp.up_proj");
    LOADQ(w_down, "mlp.down_proj");
    #undef LOADQ
    if (c->qk_norm) {
      snprintf(key, sizeof(key), "model.layers.%d.self_attn.q_norm.weight", i);
      L->q_norm = take_weight(data, key, true);
      snprintf(key, sizeof(key), "model.layers.%d.self_attn.k_norm.weight", i);
      L->k_norm = take_weight(data, key, true);
    } else {
      L->q_norm = mlx_array_new();
      L->k_norm = mlx_array_new();
    }
  }

  mlx_map_string_to_array_free(data);
  mlx_map_string_to_string_free(meta);

  /* Force materialization so subsequent GPU ops don't try to eval `Load` on GPU. */
  eval_weight(W->tok_embed);
  eval_weight(W->norm);
  eval_qlinear(&W->lm_head);
  for (int i = 0; i < c->n_layers; i++) {
    LayerW* L = &W->layers[i];
    eval_weight(L->attn_norm); eval_weight(L->ffn_norm);
    eval_qlinear(&L->wq); eval_qlinear(&L->wk);
    eval_qlinear(&L->wv); eval_qlinear(&L->wo);
    eval_qlinear(&L->w_gate); eval_qlinear(&L->w_up); eval_qlinear(&L->w_down);
    if (L->q_norm.ctx) eval_weight(L->q_norm);
    if (L->k_norm.ctx) eval_weight(L->k_norm);
  }
}

static void free_weights(Weights* W, const smlx_config* c) {
  for (int i = 0; i < c->n_layers; i++) {
    LayerW* L = &W->layers[i];
    mlx_array_free(L->attn_norm); mlx_array_free(L->ffn_norm);
    free_qlinear(&L->wq); free_qlinear(&L->wk);
    free_qlinear(&L->wv); free_qlinear(&L->wo);
    free_qlinear(&L->w_gate); free_qlinear(&L->w_up); free_qlinear(&L->w_down);
    mlx_array_free(L->q_norm); mlx_array_free(L->k_norm);
  }
  free(W->layers);
  mlx_array_free(W->tok_embed);
  mlx_array_free(W->norm);
  free_qlinear(&W->lm_head);
  mlx_array_free(W->rope_freqs);
  free(W->rope_freqs_buf);
}

/* ---------- small op helpers ---------- */

static mlx_array reshape(mlx_array x, const int* shape, int n, mlx_stream s) {
  mlx_array r = mlx_array_new();
  CHECK(mlx_reshape(&r, x, shape, (size_t)n, s));
  mlx_array_free(x);
  return r;
}

static mlx_array transpose_axes(mlx_array x, const int* axes, int n, mlx_stream s) {
  mlx_array r = mlx_array_new();
  CHECK(mlx_transpose_axes(&r, x, axes, (size_t)n, s));
  mlx_array_free(x);
  return r;
}

static mlx_array matmul(mlx_array a, mlx_array b, mlx_stream s) {
  mlx_array r = mlx_array_new();
  CHECK(mlx_matmul(&r, a, b, s));
  mlx_array_free(a);
  return r;
}

static mlx_array add(mlx_array a, mlx_array b, mlx_stream s) {
  mlx_array r = mlx_array_new();
  CHECK(mlx_add(&r, a, b, s));
  mlx_array_free(a); mlx_array_free(b);
  return r;
}

static mlx_array mul(mlx_array a, mlx_array b, mlx_stream s) {
  mlx_array r = mlx_array_new();
  CHECK(mlx_multiply(&r, a, b, s));
  mlx_array_free(a); mlx_array_free(b);
  return r;
}

static mlx_array sigmoid(mlx_array a, mlx_stream s) {
  mlx_array r = mlx_array_new();
  CHECK(mlx_sigmoid(&r, a, s));
  mlx_array_free(a);
  return r;
}

static mlx_array rms_norm(mlx_array x, mlx_array w, float eps, mlx_stream s) {
  mlx_array r = mlx_array_new();
  CHECK(mlx_fast_rms_norm(&r, x, w, eps, s));
  mlx_array_free(x);
  return r;
}

static mlx_array rope(mlx_array x, int dims, float base, int offset,
                      mlx_array freqs, bool use_freqs, mlx_stream s) {
  mlx_array r = mlx_array_new();
  mlx_optional_float b = {
    .value = use_freqs ? 0.0f : base,
    .has_value = !use_freqs,
  };
  mlx_array f = use_freqs ? freqs : (mlx_array){ 0 };
  CHECK(mlx_fast_rope(&r, x, dims, /*traditional=*/false, b, 1.0f, offset, f, s));
  mlx_array_free(x);
  return r;
}

static mlx_array sdpa(mlx_array q, mlx_array k, mlx_array v, float scale,
                      const char* mask_mode, mlx_stream s) {
  mlx_array r = mlx_array_new();
  mlx_array null_arr = { 0 };
  CHECK(mlx_fast_scaled_dot_product_attention(
      &r, q, k, v, scale, mask_mode, null_arr, null_arr, s));
  mlx_array_free(q); mlx_array_free(k); mlx_array_free(v);
  return r;
}

static mlx_array concat2(mlx_array a, mlx_array b, int axis, mlx_stream s) {
  mlx_array arrs[2] = { a, b };
  mlx_vector_array v = mlx_vector_array_new_data(arrs, 2);
  mlx_array r = mlx_array_new();
  CHECK(mlx_concatenate_axis(&r, v, axis, s));
  mlx_vector_array_free(v);
  mlx_array_free(a); mlx_array_free(b);
  return r;
}

static mlx_array take_axis0(mlx_array a, mlx_array idx, mlx_stream s) {
  mlx_array r = mlx_array_new();
  CHECK(mlx_take_axis(&r, a, idx, 0, s));
  return r;
}

/* Last position along axis=1: x[:, -1:, :]. */
static mlx_array last_token(mlx_array x, int seq_len, mlx_stream s) {
  size_t nd = mlx_array_ndim(x);
  const int* shape = mlx_array_shape(x);
  int start[8], stop[8], stride[8];
  for (size_t i = 0; i < nd; i++) { start[i] = 0; stop[i] = shape[i]; stride[i] = 1; }
  start[1] = seq_len - 1;
  stop[1] = seq_len;
  mlx_array r = mlx_array_new();
  CHECK(mlx_slice(&r, x, start, nd, stop, nd, stride, nd, s));
  mlx_array_free(x);
  return r;
}

/* ---------- sampling ---------- */
/* smlx_sampling struct is defined in smlx.h. */

/* Argmax over [1, 1, vocab] -> uint32 id. Forces eval. */
static uint32_t argmax_id(mlx_array logits, mlx_stream s) {
  mlx_array idx = mlx_array_new();
  CHECK(mlx_argmax_axis(&idx, logits, -1, false, s));
  mlx_array_free(logits);
  int sh[1] = {1};
  mlx_array flat = mlx_array_new();
  CHECK(mlx_reshape(&flat, idx, sh, 1, s));
  mlx_array_free(idx);
  CHECK(mlx_array_eval(flat));
  uint32_t out;
  CHECK(mlx_array_item_uint32(&out, flat));
  mlx_array_free(flat);
  return out;
}

/* Sample one id from logits[1,1,vocab]. Consumes logits. */
static uint32_t sample_id(mlx_array logits, const smlx_config* cfg, const smlx_sampling* p,
                          uint64_t step_seed, mlx_stream s) {
  if (p->temperature == 0.0f) return argmax_id(logits, s);

  /* temperature scale */
  mlx_array t = mlx_array_new_float(p->temperature);
  mlx_array sc = mlx_array_new();
  CHECK(mlx_divide(&sc, logits, t, s));
  mlx_array_free(t); mlx_array_free(logits);
  logits = sc;

  const int V = cfg->vocab_size;

  /* top-k: keep only the K largest; threshold = K-th largest value. */
  if (p->top_k > 0 && p->top_k < V) {
    /* mlx_partition_axis places the (kth)-smallest in slot kth. We want the
     * (V - top_k)-th smallest = top_k-th largest. */
    mlx_array part = mlx_array_new();
    CHECK(mlx_partition_axis(&part, logits, V - p->top_k, -1, s));
    int start[3]  = {0, 0, V - p->top_k};
    int stop[3]   = {1, 1, V - p->top_k + 1};
    int stride[3] = {1, 1, 1};
    mlx_array thr = mlx_array_new();
    CHECK(mlx_slice(&thr, part, start, 3, stop, 3, stride, 3, s));
    mlx_array_free(part);

    mlx_array mask = mlx_array_new();
    CHECK(mlx_less(&mask, logits, thr, s));
    mlx_array_free(thr);

    mlx_array neg_inf = mlx_array_new_float(-INFINITY);
    mlx_array gated = mlx_array_new();
    CHECK(mlx_where(&gated, mask, neg_inf, logits, s));
    mlx_array_free(mask); mlx_array_free(neg_inf); mlx_array_free(logits);
    logits = gated;
  }

  /* top-p (nucleus): keep smallest set whose cumulative prob exceeds p, in sorted order. */
  if (p->top_p < 1.0f && p->top_p > 0.0f) {
    /* descending sort via argsort(-logits) */
    mlx_array neg = mlx_array_new();   CHECK(mlx_negative(&neg, logits, s));
    mlx_array idx = mlx_array_new();   CHECK(mlx_argsort_axis(&idx, neg, -1, s));
    mlx_array_free(neg);

    mlx_array sorted = mlx_array_new();
    CHECK(mlx_take_along_axis(&sorted, logits, idx, -1, s));

    mlx_array probs = mlx_array_new();
    CHECK(mlx_softmax_axis(&probs, sorted, -1, /*precise=*/false, s));
    /* Exclusive cumsum so position 0 (the top token) is always included. */
    mlx_array cs = mlx_array_new();
    CHECK(mlx_cumsum(&cs, probs, -1, /*reverse=*/false, /*inclusive=*/false, s));
    mlx_array_free(probs);

    mlx_array thr = mlx_array_new_float(p->top_p);
    mlx_array mask = mlx_array_new();
    CHECK(mlx_less(&mask, cs, thr, s));
    mlx_array_free(cs); mlx_array_free(thr);

    mlx_array neg_inf = mlx_array_new_float(-INFINITY);
    mlx_array filt_sorted = mlx_array_new();
    CHECK(mlx_where(&filt_sorted, mask, sorted, neg_inf, s));
    mlx_array_free(mask); mlx_array_free(neg_inf); mlx_array_free(sorted);

    /* Scatter back to original positions: inv = argsort(idx). */
    mlx_array inv = mlx_array_new();
    CHECK(mlx_argsort_axis(&inv, idx, -1, s));
    mlx_array_free(idx);

    mlx_array unsorted = mlx_array_new();
    CHECK(mlx_take_along_axis(&unsorted, filt_sorted, inv, -1, s));
    mlx_array_free(filt_sorted); mlx_array_free(inv);
    mlx_array_free(logits);
    logits = unsorted;
  }

  /* sample */
  mlx_array key = mlx_array_new();
  CHECK(mlx_random_key(&key, step_seed));
  mlx_array sampled = mlx_array_new();
  CHECK(mlx_random_categorical(&sampled, logits, -1, key, s));
  mlx_array_free(logits); mlx_array_free(key);

  /* sampled has shape [1, 1] int32. */
  int sh[1] = {1};
  mlx_array flat = mlx_array_new();
  CHECK(mlx_reshape(&flat, sampled, sh, 1, s));
  mlx_array_free(sampled);
  CHECK(mlx_array_eval(flat));
  /* categorical returns int32, not uint32. */
  int32_t out_i;
  CHECK(mlx_array_item_int32(&out_i, flat));
  mlx_array_free(flat);
  return (uint32_t)out_i;
}

/* ---------- forward ---------- */

/* Preallocated KV cache. Per layer we keep [1, n_kv_heads, capacity, head_dim]
 * buffers that grow in chunks (CHUNK tokens at a time) and write new K/V into
 * the [offset : offset+T_new] slice via slice_update. This avoids the
 * per-step O(T) concat-and-copy of the earlier implementation. */
#define KV_CHUNK 256

typedef struct {
  mlx_array* keys;     /* per-layer [1, n_kv_heads, capacity[i], head_dim] */
  mlx_array* values;
  int*       capacity; /* per-layer (grows monotonically) */
  bool*      primed;
  int offset;          /* filled length, shared across layers */
  int n_kv_heads;
  int head_dim;
} KVCache;

static KVCache cache_new(int n_layers, int n_kv_heads, int head_dim) {
  KVCache c;
  c.keys     = calloc(n_layers, sizeof(mlx_array));
  c.values   = calloc(n_layers, sizeof(mlx_array));
  c.capacity = calloc(n_layers, sizeof(int));
  c.primed   = calloc(n_layers, sizeof(bool));
  c.offset = 0;
  c.n_kv_heads = n_kv_heads;
  c.head_dim   = head_dim;
  return c;
}

static void cache_free(KVCache* c, int n_layers) {
  for (int i = 0; i < n_layers; i++) {
    if (c->primed[i]) { mlx_array_free(c->keys[i]); mlx_array_free(c->values[i]); }
  }
  free(c->keys); free(c->values); free(c->capacity); free(c->primed);
}

/* Grow layer i's K/V buffers to at least `needed` columns, in CHUNK steps.
 * Preserves any existing content. `ref_for_dtype` provides the dtype. */
static void cache_grow(KVCache* c, int layer, int needed, mlx_array ref_for_dtype,
                       mlx_stream s) {
  int new_cap = ((needed + KV_CHUNK - 1) / KV_CHUNK) * KV_CHUNK;
  mlx_dtype dt = mlx_array_dtype(ref_for_dtype);
  int shape[4] = {1, c->n_kv_heads, new_cap, c->head_dim};

  mlx_array nk = mlx_array_new(); CHECK(mlx_zeros(&nk, shape, 4, dt, s));
  mlx_array nv = mlx_array_new(); CHECK(mlx_zeros(&nv, shape, 4, dt, s));

  if (c->primed[layer]) {
    int start[4]  = {0, 0, 0, 0};
    int stop[4]   = {1, c->n_kv_heads, c->capacity[layer], c->head_dim};
    int stride[4] = {1, 1, 1, 1};
    mlx_array tmp = mlx_array_new();
    CHECK(mlx_slice_update(&tmp, nk, c->keys[layer],   start, 4, stop, 4, stride, 4, s));
    mlx_array_free(nk); nk = tmp;
    tmp = mlx_array_new();
    CHECK(mlx_slice_update(&tmp, nv, c->values[layer], start, 4, stop, 4, stride, 4, s));
    mlx_array_free(nv); nv = tmp;
    mlx_array_free(c->keys[layer]);
    mlx_array_free(c->values[layer]);
  }
  c->keys[layer]   = nk;
  c->values[layer] = nv;
  c->capacity[layer] = new_cap;
  c->primed[layer] = true;
}

/* Write new_k, new_v (shape [1, H, T_new, D]) at offset, then return the
 * filled window [0 : offset+T_new] as out_k, out_v (fresh handles owned by
 * caller). Consumes new_k, new_v. */
static void cache_update(KVCache* c, int layer, mlx_array new_k, mlx_array new_v,
                         int T_new, mlx_array* out_k, mlx_array* out_v, mlx_stream s) {
  int needed = c->offset + T_new;
  if (!c->primed[layer] || needed > c->capacity[layer]) {
    cache_grow(c, layer, needed, new_k, s);
  }
  int start[4]  = {0, 0, c->offset, 0};
  int stop[4]   = {1, c->n_kv_heads, c->offset + T_new, c->head_dim};
  int stride[4] = {1, 1, 1, 1};
  mlx_array nk = mlx_array_new();
  mlx_array nv = mlx_array_new();
  CHECK(mlx_slice_update(&nk, c->keys[layer],   new_k, start, 4, stop, 4, stride, 4, s));
  CHECK(mlx_slice_update(&nv, c->values[layer], new_v, start, 4, stop, 4, stride, 4, s));
  mlx_array_free(c->keys[layer]);
  mlx_array_free(c->values[layer]);
  c->keys[layer]   = nk;
  c->values[layer] = nv;
  mlx_array_free(new_k);
  mlx_array_free(new_v);

  int wstart[4] = {0, 0, 0, 0};
  int wstop[4]  = {1, c->n_kv_heads, c->offset + T_new, c->head_dim};
  CHECK(mlx_slice(out_k, c->keys[layer],   wstart, 4, wstop, 4, stride, 4, s));
  CHECK(mlx_slice(out_v, c->values[layer], wstart, 4, wstop, 4, stride, 4, s));
}

/* Run forward on tokens [B=1, T] -> sampled token id (uint32). */
static uint32_t forward(const smlx_config* cfg, Weights* W, KVCache* kv,
                        const int32_t* tok_ids, int T,
                        const smlx_sampling* samp, mlx_stream s) {
  /* Build token id array [1, T] -> embed -> [1, T, dim] */
  int tok_shape[2] = {1, T};
  mlx_array tokens = mlx_array_new_data(tok_ids, tok_shape, 2, MLX_INT32);

  /* Embedding via take(axis=0). Need to reshape to [T] for axis-0 take, then reshape back. */
  int t_shape[1] = {T};
  mlx_array tokens_flat = mlx_array_new();
  CHECK(mlx_reshape(&tokens_flat, tokens, t_shape, 1, s));
  mlx_array_free(tokens);

  mlx_array x = take_axis0(W->tok_embed, tokens_flat, s);  /* [T, dim] */
  mlx_array_free(tokens_flat);
  int xshape[3] = {1, T, cfg->dim};
  x = reshape(x, xshape, 3, s);  /* [1, T, dim] */

  const int n_h   = cfg->n_heads;
  const int n_kvh = cfg->n_kv_heads;
  const int d_h   = cfg->head_dim;
  const float scale = 1.0f / sqrtf((float)d_h);
  const int qkv_perm[4] = {0, 2, 1, 3};

  for (int i = 0; i < cfg->n_layers; i++) {
    LayerW* L = &W->layers[i];

    /* --- attention --- */
    mlx_array h = mlx_array_new();
    CHECK(mlx_fast_rms_norm(&h, x, L->attn_norm, cfg->norm_eps, s));

    /* Project Q/K/V. qlinear_apply consumes its input `x`, so reuse h once and
     * for the other two start from a fresh handle pointing at the same data. */
    mlx_array h2 = mlx_array_new(); CHECK(mlx_array_set(&h2, h));
    mlx_array h3 = mlx_array_new(); CHECK(mlx_array_set(&h3, h));
    mlx_array q = qlinear_apply(&L->wq, h,  cfg, s);
    mlx_array k = qlinear_apply(&L->wk, h2, cfg, s);
    mlx_array v = qlinear_apply(&L->wv, h3, cfg, s);

    int q_shape[4] = {1, T, n_h,   d_h};
    int k_shape[4] = {1, T, n_kvh, d_h};
    q = reshape(q, q_shape, 4, s);
    k = reshape(k, k_shape, 4, s);
    v = reshape(v, k_shape, 4, s);

    /* Qwen3 QK-Norm: per-head RMSNorm over the head_dim axis, on [B,T,H,D]
     * (last axis = head_dim), applied before the transpose and RoPE. */
    if (cfg->qk_norm) {
      mlx_array qn = mlx_array_new();
      CHECK(mlx_fast_rms_norm(&qn, q, L->q_norm, cfg->norm_eps, s));
      mlx_array_free(q); q = qn;
      mlx_array kn = mlx_array_new();
      CHECK(mlx_fast_rms_norm(&kn, k, L->k_norm, cfg->norm_eps, s));
      mlx_array_free(k); k = kn;
    }

    /* [B, T, H, D] -> [B, H, T, D] */
    q = transpose_axes(q, qkv_perm, 4, s);
    k = transpose_axes(k, qkv_perm, 4, s);
    v = transpose_axes(v, qkv_perm, 4, s);

    /* RoPE on Q and K (in place of position embedding). */
    q = rope(q, d_h, cfg->rope_theta, kv->offset, W->rope_freqs, W->use_rope_freqs, s);
    k = rope(k, d_h, cfg->rope_theta, kv->offset, W->rope_freqs, W->use_rope_freqs, s);

    /* Update preallocated KV cache slot at [offset:offset+T] and get back the
     * filled window [0:offset+T] for attention. */
    mlx_array k_win = mlx_array_new();
    mlx_array v_win = mlx_array_new();
    cache_update(kv, i, k, v, T, &k_win, &v_win, s);

    const char* mask = (T > 1) ? "causal" : "";
    mlx_array attn = sdpa(q, k_win, v_win, scale, mask, s);  /* [1, n_h, T, d_h] */

    int back_perm[4] = {0, 2, 1, 3};
    attn = transpose_axes(attn, back_perm, 4, s);    /* [1, T, n_h, d_h] */
    int merged[3] = {1, T, n_h * d_h};
    attn = reshape(attn, merged, 3, s);

    mlx_array attn_out = qlinear_apply(&L->wo, attn, cfg, s);
    x = add(x, attn_out, s);

    /* --- mlp (SwiGLU) --- */
    mlx_array hn = mlx_array_new();
    CHECK(mlx_fast_rms_norm(&hn, x, L->ffn_norm, cfg->norm_eps, s));
    mlx_array hn2 = mlx_array_new(); CHECK(mlx_array_set(&hn2, hn));
    mlx_array g = qlinear_apply(&L->w_gate, hn,  cfg, s);
    mlx_array u = qlinear_apply(&L->w_up,   hn2, cfg, s);

    /* silu(g) = g * sigmoid(g) */
    mlx_array g_sig = mlx_array_new();
    CHECK(mlx_sigmoid(&g_sig, g, s));
    mlx_array silu_g = mul(g, g_sig, s);
    mlx_array gated = mul(silu_g, u, s);

    mlx_array mlp_out = qlinear_apply(&L->w_down, gated, cfg, s);
    x = add(x, mlp_out, s);
  }

  /* Final norm + lm head on last position only. */
  mlx_array xn = mlx_array_new();
  CHECK(mlx_fast_rms_norm(&xn, x, W->norm, cfg->norm_eps, s));
  mlx_array_free(x);

  xn = last_token(xn, T, s);  /* [1, 1, dim] */
  mlx_array logits = qlinear_apply(&W->lm_head, xn, cfg, s);  /* [1, 1, vocab] */

  uint32_t out = sample_id(logits, cfg, samp, samp->seed + (uint64_t)kv->offset, s);
  kv->offset += T;
  return out;
}

/* ---------- public API ---------- */

struct smlx_model {
  smlx_config cfg;
  Weights     w;
  mlx_stream  gpu;
};

struct smlx_session {
  smlx_model* m;
  KVCache     kv;
};

smlx_model* smlx_model_load(smlx_config* cfg, const char* safetensors_path) {
  smlx_model* m = calloc(1, sizeof(*m));
  m->cfg = *cfg;
  mlx_stream cpu = mlx_default_cpu_stream_new();
  m->gpu = mlx_default_gpu_stream_new();
  load_weights(&m->w, &m->cfg, safetensors_path, cpu);
  build_rope_freqs(&m->w, &m->cfg);
  mlx_stream_free(cpu);
  /* Publish auto-detected config back to caller. */
  *cfg = m->cfg;
  return m;
}

void smlx_model_free(smlx_model* m) {
  if (!m) return;
  free_weights(&m->w, &m->cfg);
  mlx_stream_free(m->gpu);
  free(m);
}

smlx_session* smlx_session_new(smlx_model* m) {
  smlx_session* s = calloc(1, sizeof(*s));
  s->m = m;
  s->kv = cache_new(m->cfg.n_layers, m->cfg.n_kv_heads, m->cfg.head_dim);
  return s;
}

void smlx_session_free(smlx_session* s) {
  if (!s) return;
  cache_free(&s->kv, s->m->cfg.n_layers);
  free(s);
}

int smlx_session_offset(const smlx_session* s) {
  return s->kv.offset;
}

uint32_t smlx_generate(smlx_session* s, const int32_t* tokens, int n_tokens,
                       const smlx_sampling* samp) {
  return forward(&s->m->cfg, &s->m->w, &s->kv, tokens, n_tokens, samp, s->m->gpu);
}
