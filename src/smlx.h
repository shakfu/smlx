/*
 * libsmlx -- Llama-style transformer inference on top of mlx-c.
 *
 * Public C API. Internals (KV cache, quantized linears, RoPE freqs) are
 * hidden behind opaque handles in smlx.c.
 *
 * Lifecycle:
 *   smlx_config cfg;
 *   smlx_config_load(&cfg, "smlx.config.txt");
 *   smlx_model* m = smlx_model_load(&cfg, "model.safetensors");
 *   smlx_session* sess = smlx_session_new(m);
 *   uint32_t id = smlx_generate(sess, prompt_ids, n_prompt, &samp);
 *   // ... loop calling smlx_generate(sess, &one, 1, &samp) ...
 *   smlx_session_free(sess);
 *   smlx_model_free(m);
 */

#ifndef SMLX_H
#define SMLX_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct smlx_config {
  int   n_layers, dim, n_heads, n_kv_heads, head_dim, hidden_dim, vocab_size;
  float rope_theta, norm_eps;
  /* Llama-3 RoPE scaling. Set rope_original_max_pos > 0 to enable. */
  float rope_factor, rope_low_freq_factor, rope_high_freq_factor;
  int   rope_original_max_pos;
  /* Quantization. Filled by smlx_model_load if the safetensors are quantized. */
  bool  quantized;
  int   q_bits;
  int   q_group_size;
  /* Architecture variants, auto-detected at load:
   *   attn_qkv_bias -- Q/K/V projections have a bias (Qwen 2.x)
   *   qk_norm       -- per-head RMSNorm on Q and K before RoPE (Qwen 3) */
  bool  attn_qkv_bias;
  bool  qk_norm;
} smlx_config;

typedef struct smlx_sampling {
  float    temperature;  /* 0 = greedy/argmax; top_k/top_p ignored. */
  int      top_k;        /* 0 = disabled */
  float    top_p;        /* >=1.0 = disabled */
  uint64_t seed;         /* per-step key derived as seed + session_offset */
} smlx_sampling;

typedef struct smlx_model   smlx_model;
typedef struct smlx_session smlx_session;

/* Parse a "key=value\n" config file. Returns 0 on success, -1 on failure. */
int smlx_config_load(smlx_config* out, const char* path);

/* Load weights. Quantization is auto-detected; fields in `cfg` may be updated.
 * Returns NULL on failure. */
smlx_model* smlx_model_load(smlx_config* cfg, const char* safetensors_path);
void        smlx_model_free(smlx_model* m);

/* New inference session (independent KV cache). */
smlx_session* smlx_session_new(smlx_model* m);
void          smlx_session_free(smlx_session* s);
int           smlx_session_offset(const smlx_session* s);

/* Forward on [B=1, n_tokens], sample one token id. Advances session offset. */
uint32_t smlx_generate(smlx_session* s, const int32_t* tokens, int n_tokens,
                       const smlx_sampling* samp);

#ifdef __cplusplus
}
#endif

#endif /* SMLX_H */
