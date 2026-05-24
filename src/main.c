/*
 * smlx CLI -- thin wrapper around libsmlx.
 *
 * Reads a config file and safetensors weights, then runs a prefill +
 * one-token-per-line decode loop. Sampling and EOS configured via env vars.
 *
 * Usage:
 *   smlx <config.txt> <weights.safetensors> <max_new> [<prompt_id> ...]
 * If prompt ids are not on argv, they are read from stdin (whitespace-separated).
 *
 * Env:
 *   SMLX_TEMP, SMLX_TOP_K, SMLX_TOP_P, SMLX_SEED -- sampling
 *   SMLX_EOS -- comma-separated stop ids (break decode when sampled)
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "smlx.h"

static double now_s(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void usage(void) {
  fprintf(stderr,
    "usage: smlx <config.txt> <weights.safetensors> <max_new> [<prompt_id> ...]\n"
    "  if no prompt ids are given on the command line, they are read as\n"
    "  whitespace-separated integers from stdin.\n");
  exit(2);
}

static int32_t* read_ids_stream(FILE* f, int* n_out) {
  size_t cap = 64, n = 0;
  int32_t* buf = malloc(sizeof(int32_t) * cap);
  long v;
  while (fscanf(f, " %ld", &v) == 1) {
    if (n == cap) { cap *= 2; buf = realloc(buf, sizeof(int32_t) * cap); }
    buf[n++] = (int32_t)v;
  }
  *n_out = (int)n;
  return buf;
}

int main(int argc, char** argv) {
  if (argc < 4) usage();
  const char* cfg_path = argv[1];
  const char* w_path   = argv[2];
  int max_new          = atoi(argv[3]);

  int n_prompt;
  int32_t* prompt;
  if (argc > 4) {
    n_prompt = argc - 4;
    prompt = malloc(sizeof(int32_t) * n_prompt);
    for (int i = 0; i < n_prompt; i++) prompt[i] = (int32_t)atoi(argv[4 + i]);
  } else {
    prompt = read_ids_stream(stdin, &n_prompt);
    if (n_prompt == 0) { fprintf(stderr, "no prompt ids on stdin\n"); return 2; }
  }

  smlx_config cfg;
  if (smlx_config_load(&cfg, cfg_path) != 0) return 1;

  smlx_model* model = smlx_model_load(&cfg, w_path);
  smlx_session* sess = smlx_session_new(model);

  smlx_sampling samp = {0};
  const char* e;
  if ((e = getenv("SMLX_TEMP")))  samp.temperature = (float)atof(e);
  if ((e = getenv("SMLX_TOP_K"))) samp.top_k       = atoi(e);
  if ((e = getenv("SMLX_TOP_P"))) samp.top_p       = (float)atof(e);
  else                            samp.top_p       = 1.0f;
  if ((e = getenv("SMLX_SEED")))  samp.seed        = (uint64_t)strtoull(e, NULL, 10);

  int eos[16]; int n_eos = 0;
  if ((e = getenv("SMLX_EOS"))) {
    char* dup = strdup(e);
    for (char* tok = strtok(dup, ","); tok && n_eos < 16; tok = strtok(NULL, ",")) {
      eos[n_eos++] = atoi(tok);
    }
    free(dup);
  }
  #define IS_EOS(id) ({ bool _r = false; \
    for (int _i = 0; _i < n_eos; _i++) if ((int)(id) == eos[_i]) { _r = true; break; } \
    _r; })

  double t0 = now_s();
  uint32_t next = smlx_generate(sess, prompt, n_prompt, &samp);
  double t1 = now_s();
  printf("%u\n", next);
  fflush(stdout);

  int decoded = 1;
  if (!IS_EOS(next)) {
    for (int i = 1; i < max_new; i++) {
      int32_t one = (int32_t)next;
      next = smlx_generate(sess, &one, 1, &samp);
      printf("%u\n", next);
      fflush(stdout);
      decoded++;
      if (IS_EOS(next)) break;
    }
  }
  double t2 = now_s();

  double prefill_s = t1 - t0, decode_s = t2 - t1;
  int dec_after = decoded - 1;
  fprintf(stderr,
    "[smlx] prefill: %d tok in %.3fs = %.1f tok/s | "
    "decode: %d tok in %.3fs = %.1f tok/s\n",
    n_prompt, prefill_s, prefill_s > 0 ? n_prompt / prefill_s : 0.0,
    dec_after, decode_s,
    decode_s > 0 && dec_after > 0 ? dec_after / decode_s : 0.0);

  smlx_session_free(sess);
  smlx_model_free(model);
  free(prompt);
  return 0;
}
