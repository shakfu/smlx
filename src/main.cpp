/*
 * smlx -- unified CLI on top of libsmlx.
 *
 *   smlx chat -m <model_dir> [-p "<prompt>"] [options]   text chat (no Python)
 *   smlx ids  <config> <weights> <max> [<id> ...]        raw token ids in/out
 *
 * `chat` tokenizes + renders the model's chat template in-process (tokenizers-cpp
 * + minja) and streams text. `ids` is the bare engine: token ids in, ids out,
 * sampling/EOS via env vars. Both share libsmlx.
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "smlx.h"
#include "tokenizers_cpp.h"
#include "minja/chat-template.hpp"
#include "nlohmann/json.hpp"

using json = nlohmann::ordered_json;

static double now_s() {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { fprintf(stderr, "smlx: cannot open %s\n", path.c_str()); exit(1); }
  std::stringstream ss; ss << f.rdbuf();
  return ss.str();
}

static bool file_exists(const std::string& p) { std::ifstream f(p); return f.good(); }

/* ===================== smlx chat ===================== */

static std::string token_str(const json& j, const char* key) {
  if (!j.contains(key) || j[key].is_null()) return "";
  const auto& v = j[key];
  if (v.is_string()) return v.get<std::string>();
  if (v.is_object() && v.contains("content")) return v["content"].get<std::string>();
  return "";
}

static void chat_usage() {
  fprintf(stderr,
    "usage: smlx chat -m <model_dir> [-p \"<prompt>\"] [options]\n"
    "  -m, --model DIR       model directory (required)\n"
    "  -p, --prompt TEXT     single-shot; omit for interactive chat\n"
    "  --max N               answer-token budget (default 512)\n"
    "  --think-budget N      max thinking tokens before </think> is forced\n"
    "                        (default 4096; 0 disables). Total = think-budget + max.\n"
    "  --temp F --top-k K --top-p F --seed S    sampling\n"
    "  --raw                 skip chat template; feed prompt verbatim\n"
    "  --no-think            disable Qwen3 thinking mode\n");
  exit(2);
}

static int cmd_chat(int argc, char** argv) {
  std::string model_dir, prompt;
  bool has_prompt = false, raw = false, no_think = false;
  int  max_new = -1, top_k = 0, think_budget = 4096;  /* max_new<0 => auto */
  float temp = 0.0f, top_p = 1.0f;
  uint64_t seed = 0;

  for (int i = 2; i < argc; i++) {  /* argv[1] == "chat" */
    std::string a = argv[i];
    auto next = [&](const char* what) -> char* {
      if (i + 1 >= argc) { fprintf(stderr, "smlx chat: %s needs a value\n", what); chat_usage(); }
      return argv[++i];
    };
    if      (a == "-m" || a == "--model")  model_dir = next("-m");
    else if (a == "-p" || a == "--prompt") { prompt = next("-p"); has_prompt = true; }
    else if (a == "--max")     max_new = atoi(next("--max"));
    else if (a == "--temp")    temp = (float)atof(next("--temp"));
    else if (a == "--top-k")   top_k = atoi(next("--top-k"));
    else if (a == "--top-p")   top_p = (float)atof(next("--top-p"));
    else if (a == "--seed")    seed = strtoull(next("--seed"), nullptr, 10);
    else if (a == "--raw")     raw = true;
    else if (a == "--no-think") no_think = true;
    else if (a == "--think-budget") think_budget = atoi(next("--think-budget"));
    else if (a == "-h" || a == "--help") chat_usage();
    else { fprintf(stderr, "smlx chat: unknown arg '%s'\n", a.c_str()); chat_usage(); }
  }
  if (model_dir.empty()) chat_usage();
  if (max_new < 0) max_new = 512;
  const int total_cap = (think_budget > 0 ? think_budget : 0) + max_new;

  const std::string cfg_path  = model_dir + "/smlx.config.txt";
  const std::string weights   = model_dir + "/model.safetensors";
  const std::string tok_path  = model_dir + "/tokenizer.json";
  const std::string tcfg_path = model_dir + "/tokenizer_config.json";
  const std::string gcfg_path = model_dir + "/generation_config.json";

  smlx_config cfg;
  if (smlx_config_load(&cfg, cfg_path.c_str()) != 0) return 1;
  smlx_model* model = smlx_model_load(&cfg, weights.c_str());
  auto tok = tokenizers::Tokenizer::FromBlobJSON(read_file(tok_path));

  std::string tmpl, bos, eos_tok;
  if (!raw) {
    json tcfg = json::parse(read_file(tcfg_path));
    bos = token_str(tcfg, "bos_token");
    eos_tok = token_str(tcfg, "eos_token");
    if (tcfg.contains("chat_template")) {
      const auto& ct = tcfg["chat_template"];
      if (ct.is_string()) tmpl = ct.get<std::string>();
      else if (ct.is_array() && !ct.empty()) {
        for (const auto& e : ct)
          if (e.value("name", "") == "default") tmpl = e.value("template", "");
        if (tmpl.empty()) tmpl = ct[0].value("template", "");
      }
    }
    if (tmpl.empty() && file_exists(model_dir + "/chat_template.jinja"))
      tmpl = read_file(model_dir + "/chat_template.jinja");
    if (tmpl.empty()) { fprintf(stderr, "smlx chat: no chat_template; use --raw\n"); return 1; }
  }

  std::set<int> eos;
  {
    if (!eos_tok.empty()) { int id = tok->TokenToId(eos_tok); if (id >= 0) eos.insert(id); }
    if (file_exists(gcfg_path)) {
      json g = json::parse(read_file(gcfg_path));
      if (g.contains("eos_token_id")) {
        const auto& ei = g["eos_token_id"];
        if (ei.is_number_integer()) eos.insert(ei.get<int>());
        else if (ei.is_array()) for (const auto& x : ei) eos.insert(x.get<int>());
      }
    }
    if (eos.empty()) fprintf(stderr, "smlx chat: warning, no EOS token found\n");
  }

  smlx_sampling samp{};
  samp.temperature = temp; samp.top_k = top_k; samp.top_p = top_p; samp.seed = seed;

  std::unique_ptr<minja::chat_template> chat;
  if (!raw) chat = std::make_unique<minja::chat_template>(tmpl, bos, eos_tok);

  auto generate = [&](const json& messages, const std::string& raw_text) -> std::string {
    std::string text;
    if (raw) {
      text = raw_text;
    } else {
      minja::chat_template_inputs in;
      in.messages = messages;
      in.add_generation_prompt = true;
      if (no_think) in.extra_context = json{{"enable_thinking", false}};
      text = chat->apply(in);
    }
    std::vector<int32_t> ids = tok->Encode(text);

    /* Trim a trailing run of U+FFFD (incomplete multi-byte char) so streaming
     * doesn't print replacement boxes mid-emoji. */
    auto valid_end = [](const std::string& s) -> size_t {
      size_t n = s.size();
      while (n >= 3 && (unsigned char)s[n-3] == 0xEF &&
             (unsigned char)s[n-2] == 0xBF && (unsigned char)s[n-1] == 0xBD)
        n -= 3;
      return n;
    };
    std::vector<int32_t> close_ids = tok->Encode("\n</think>\n\n");

    smlx_session* sess = smlx_session_new(model);
    std::vector<int32_t> out;
    std::string last;
    auto emit = [&]() {
      std::string cur = tok->Decode(out);
      size_t ve = valid_end(cur);
      if (ve > last.size()) {
        fwrite(cur.data() + last.size(), 1, ve - last.size(), stdout);
        fflush(stdout);
        last.assign(cur.data(), ve);
      }
    };

    std::vector<int32_t> next_in = ids;
    bool forced = false;
    for (int produced = 0; produced < total_cap; produced++) {
      uint32_t next = smlx_generate(sess, next_in.data(), (int)next_in.size(), &samp);
      if (eos.count((int)next)) break;
      out.push_back((int32_t)next);
      emit();
      next_in.assign(1, (int32_t)next);

      /* Thinking budget: force </think> if opened but not closed within budget.
       * No-op when no <think> appears (non-thinking models / --no-think). */
      if (!forced && think_budget > 0) {
        const std::string& cur = last;
        bool has_open  = cur.find("<think>")  != std::string::npos;
        bool has_close = cur.find("</think>") != std::string::npos;
        if (has_open && !has_close && (int)out.size() >= think_budget) {
          next_in.insert(next_in.end(), close_ids.begin(), close_ids.end());
          out.insert(out.end(), close_ids.begin(), close_ids.end());
          emit();
          forced = true;
        }
      }
    }
    smlx_session_free(sess);
    printf("\n");
    return tok->Decode(out);
  };

  if (has_prompt) {
    json messages = json::array();
    messages.push_back({{"role", "user"}, {"content", prompt}});
    generate(messages, prompt);
  } else {
    fprintf(stderr, "smlx chat (%s). Ctrl-D or /exit to quit.\n", model_dir.c_str());
    json messages = json::array();
    std::string line;
    while (true) {
      fprintf(stderr, "\n> "); fflush(stderr);
      if (!std::getline(std::cin, line)) break;
      if (line == "/exit" || line == "/quit") break;
      if (line.empty()) continue;
      if (raw) { generate(json::array(), line); continue; }
      messages.push_back({{"role", "user"}, {"content", line}});
      std::string reply = generate(messages, "");
      messages.push_back({{"role", "assistant"}, {"content", reply}});
    }
    fprintf(stderr, "\n");
  }

  smlx_model_free(model);
  return 0;
}

/* ===================== smlx ids ===================== */

static void ids_usage() {
  fprintf(stderr,
    "usage: smlx ids <config.txt> <weights.safetensors> <max_new> [<id> ...]\n"
    "  prompt ids on argv, or whitespace-separated on stdin if none given.\n"
    "  env: SMLX_TEMP SMLX_TOP_K SMLX_TOP_P SMLX_SEED SMLX_EOS\n");
  exit(2);
}

static std::vector<int32_t> read_ids_stream(FILE* f) {
  std::vector<int32_t> v; long x;
  while (fscanf(f, " %ld", &x) == 1) v.push_back((int32_t)x);
  return v;
}

static int cmd_ids(int argc, char** argv) {
  if (argc < 5) ids_usage();  /* smlx ids <cfg> <weights> <max> */
  const char* cfg_path = argv[2];
  const char* w_path   = argv[3];
  int max_new          = atoi(argv[4]);

  std::vector<int32_t> prompt;
  if (argc > 5) for (int i = 5; i < argc; i++) prompt.push_back((int32_t)atoi(argv[i]));
  else          prompt = read_ids_stream(stdin);
  if (prompt.empty()) { fprintf(stderr, "smlx ids: no prompt ids\n"); return 2; }

  smlx_config cfg;
  if (smlx_config_load(&cfg, cfg_path) != 0) return 1;
  smlx_model* model = smlx_model_load(&cfg, w_path);
  smlx_session* sess = smlx_session_new(model);

  smlx_sampling samp{};
  samp.top_p = 1.0f;
  const char* e;
  if ((e = getenv("SMLX_TEMP")))  samp.temperature = (float)atof(e);
  if ((e = getenv("SMLX_TOP_K"))) samp.top_k       = atoi(e);
  if ((e = getenv("SMLX_TOP_P"))) samp.top_p       = (float)atof(e);
  if ((e = getenv("SMLX_SEED")))  samp.seed        = (uint64_t)strtoull(e, nullptr, 10);

  std::set<int> eos;
  if ((e = getenv("SMLX_EOS"))) {
    std::string s(e); size_t p = 0;
    while (p < s.size()) {
      size_t c = s.find(',', p);
      if (c == std::string::npos) c = s.size();
      if (c > p) eos.insert(atoi(s.substr(p, c - p).c_str()));
      p = c + 1;
    }
  }

  double t0 = now_s();
  uint32_t next = smlx_generate(sess, prompt.data(), (int)prompt.size(), &samp);
  double t1 = now_s();
  printf("%u\n", next); fflush(stdout);

  int decoded = 1;
  if (!eos.count((int)next)) {
    for (int i = 1; i < max_new; i++) {
      int32_t one = (int32_t)next;
      next = smlx_generate(sess, &one, 1, &samp);
      printf("%u\n", next); fflush(stdout);
      decoded++;
      if (eos.count((int)next)) break;
    }
  }
  double t2 = now_s();

  double prefill_s = t1 - t0, decode_s = t2 - t1;
  int dec_after = decoded - 1;
  fprintf(stderr,
    "[smlx] prefill: %d tok in %.3fs = %.1f tok/s | decode: %d tok in %.3fs = %.1f tok/s\n",
    (int)prompt.size(), prefill_s, prefill_s > 0 ? prompt.size() / prefill_s : 0.0,
    dec_after, decode_s, decode_s > 0 && dec_after > 0 ? dec_after / decode_s : 0.0);

  smlx_session_free(sess);
  smlx_model_free(model);
  return 0;
}

/* ===================== dispatch ===================== */

static void top_usage() {
  fprintf(stderr,
    "usage: smlx <command> [args]\n"
    "  chat   text chat (tokenizer + chat template built in)\n"
    "  ids    raw token-ids in/out (bare engine)\n"
    "run `smlx chat -h` for chat options.\n");
  exit(2);
}

int main(int argc, char** argv) {
  if (argc < 2) top_usage();
  std::string cmd = argv[1];
  if (cmd == "chat") return cmd_chat(argc, argv);
  if (cmd == "ids")  return cmd_ids(argc, argv);
  if (cmd == "-h" || cmd == "--help") top_usage();
  fprintf(stderr, "smlx: unknown command '%s'\n", cmd.c_str());
  top_usage();
  return 2;
}
