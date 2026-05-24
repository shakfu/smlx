/*
 * Minimal libsmlx usage example: argmax-decode 16 tokens starting from BOS.
 *
 * Build (after `cmake --build build -j`):
 *   cc -Isrc -o /tmp/hello_lib examples/hello_lib.c \
 *      build/libsmlx.a build/_deps/mlx-build/libmlx.a \
 *      build/thirdparty/mlx-c/libmlxc.a \
 *      -framework Metal -framework Accelerate -framework Foundation \
 *      -lc++ -lm
 *
 * Easier: link via CMake by adding this as a target in CMakeLists.txt.
 */
#include <stdio.h>
#include <stdlib.h>
#include "smlx.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <config.txt> <weights.safetensors>\n", argv[0]);
    return 2;
  }
  smlx_config cfg;
  if (smlx_config_load(&cfg, argv[1]) != 0) return 1;
  smlx_model* m = smlx_model_load(&cfg, argv[2]);
  smlx_session* s = smlx_session_new(m);

  smlx_sampling samp = { .temperature = 0.0f, .top_p = 1.0f };
  int32_t prompt[] = { 128000 };  /* <|begin_of_text|> */
  uint32_t id = smlx_generate(s, prompt, 1, &samp);
  printf("%u", id);
  for (int i = 0; i < 15; i++) {
    int32_t one = (int32_t)id;
    id = smlx_generate(s, &one, 1, &samp);
    printf(" %u", id);
  }
  printf("\n");

  smlx_session_free(s);
  smlx_model_free(m);
  return 0;
}
