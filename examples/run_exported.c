/*
 * Generate by calling a pre-exported mlx function (.mlxfn).
 *
 * No model-architecture code -- the whole forward pass lives inside the
 * exported function. This file only knows how to:
 *   - build an int32 token tensor of fixed length
 *   - call the imported function
 *   - argmax the logits at the last filled position
 *   - splice the new id and repeat
 *
 * Usage:
 *   run_exported <model.mlxfn> <T_FIXED> <n_new> <id1> [<id2> ...]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "mlx/c/mlx.h"

#define CHECK(call) do { if ((call) != 0) { \
  fprintf(stderr, "mlx call failed at %s:%d: %s\n", __FILE__, __LINE__, #call); \
  exit(1); } } while (0)

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s <model.mlxfn> <T_FIXED> <n_new> <id1> [<id2> ...]\n", argv[0]);
    return 2;
  }
  const char* mlxfn = argv[1];
  int T = atoi(argv[2]);
  int n_new = atoi(argv[3]);
  int n_prompt = argc - 4;
  if (n_prompt > T) { fprintf(stderr, "prompt longer than T_FIXED\n"); return 1; }

  mlx_stream s = mlx_default_gpu_stream_new();

  /* Load the exported function. */
  mlx_imported_function f = mlx_imported_function_new(mlxfn);

  /* Persistent token buffer: [1, T] int32, zero-padded. */
  int32_t* buf = calloc(T, sizeof(int32_t));
  for (int i = 0; i < n_prompt; i++) buf[i] = (int32_t)atoi(argv[4 + i]);

  int pos = n_prompt;
  for (int step = 0; step < n_new; step++) {
    if (pos > T) { fprintf(stderr, "exceeded T_FIXED\n"); break; }

    /* Wrap buf as an mlx_array [1, T] int32. */
    int shape[2] = {1, T};
    mlx_array tokens = mlx_array_new_data(buf, shape, 2, MLX_INT32);

    /* Apply: args is a vector containing just `tokens`. */
    mlx_vector_array args = mlx_vector_array_new_value(tokens);
    mlx_vector_array res = mlx_vector_array_new();
    CHECK(mlx_imported_function_apply(&res, f, args));

    /* res[0] = logits [1, T, V]. Slice the (pos-1)-th time step. */
    mlx_array logits = mlx_array_new();
    CHECK(mlx_vector_array_get(&logits, res, 0));

    int sh_start[3]  = {0, pos - 1, 0};
    /* stop along axis 2 (vocab) is the vocab size; we don't know it here,
     * but slice with stop = -1 isn't supported. Use the array's shape. */
    const int* lshape = mlx_array_shape(logits);
    int sh_stop[3]   = {1, pos, lshape[2]};
    int sh_stride[3] = {1, 1, 1};
    mlx_array last = mlx_array_new();
    CHECK(mlx_slice(&last, logits, sh_start, 3, sh_stop, 3, sh_stride, 3, s));

    mlx_array idx = mlx_array_new();
    CHECK(mlx_argmax_axis(&idx, last, -1, false, s));
    int flat_sh[1] = {1};
    mlx_array flat = mlx_array_new();
    CHECK(mlx_reshape(&flat, idx, flat_sh, 1, s));
    CHECK(mlx_array_eval(flat));
    uint32_t next;
    CHECK(mlx_array_item_uint32(&next, flat));

    printf("%u\n", next);
    fflush(stdout);
    buf[pos++] = (int32_t)next;

    mlx_array_free(flat); mlx_array_free(idx);
    mlx_array_free(last); mlx_array_free(logits);
    mlx_vector_array_free(res); mlx_vector_array_free(args);
    mlx_array_free(tokens);
  }

  free(buf);
  mlx_imported_function_free(f);
  mlx_stream_free(s);
  return 0;
}
