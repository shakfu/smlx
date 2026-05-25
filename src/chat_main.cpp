/*
 * smlx_chat -- fully self-contained text chat. No Python.
 *
 * Pipeline (all in-process):
 *   tokenizer.json        -> tokenizers-cpp  (text <-> ids)
 *   tokenizer_config.json -> minja           (render chat_template -> string)
 *   ids                   -> libsmlx         (generate)
 *
 * Usage:
 *   smlx_chat -m <model_dir> [-p "<prompt>"] [options]
 *     -m, --model DIR    model directory (required)
 *     -p, --prompt TEXT  single-shot prompt; if omitted, starts interactive chat
 *     --max N            max new tokens per reply (default 512)
 *     --temp F --top-k K --top-p F --seed S    sampling
 *     --raw              skip chat template; feed prompt text verbatim
 *     --no-think         disable Qwen3 thinking mode
 *
 * Interactive mode (no -p): type a message and press enter; "/exit" or EOF
 * (Ctrl-D) quits. Conversation history is kept across turns.
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { fprintf(stderr, "smlx_chat: cannot open %s\n", path.c_str()); exit(1); }
  std::stringstream ss; ss << f.rdbuf();
  return ss.str();
}

static bool file_exists(const std::string& p) { std::ifstream f(p); return f.good(); }

static std::string token_str(const json& j, const char* key) {
  if (!j.contains(key) || j[key].is_null()) return "";
  const auto& v = j[key];
  if (v.is_string()) return v.get<std::string>();
  if (v.is_object() && v.contains("content")) return v["content"].get<std::string>();
  return "";
}

static void usage(const char* prog) {
  fprintf(stderr,
    "usage: %s -m <model_dir> [-p \"<prompt>\"] [options]\n"
    "  -m, --model DIR       model directory (required)\n"
    "  -p, --prompt TEXT     single-shot; omit for interactive chat\n"
    "  --max N               answer-token budget (default 512)\n"
    "  --think-budget N      max thinking tokens before </think> is forced\n"
    "                        (default 4096; 0 disables). Total decode =\n"
    "                        think-budget + max.\n"
    "  --temp F --top-k K --top-p F --seed S    sampling\n"
    "  --raw                 skip chat template; feed prompt verbatim\n"
    "  --no-think            disable Qwen3 thinking mode\n", prog);
  exit(2);
}

int main(int argc, char** argv) {
  std::string model_dir, prompt;
  bool has_prompt = false, raw = false, no_think = false;
  int  max_new = -1, top_k = 0, think_budget = 4096;  /* max_new<0 => auto */
  float temp = 0.0f, top_p = 1.0f;
  uint64_t seed = 0;

  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&](const char* what) -> char* {
      if (i + 1 >= argc) { fprintf(stderr, "smlx_chat: %s needs a value\n", what); usage(argv[0]); }
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
    else if (a == "-h" || a == "--help") usage(argv[0]);
    else { fprintf(stderr, "smlx_chat: unknown arg '%s'\n", a.c_str()); usage(argv[0]); }
  }
  if (model_dir.empty()) usage(argv[0]);
  /* `--max` is the answer budget; total decode = think_budget + answer. When
   * not set explicitly, default the answer budget to 512. */
  if (max_new < 0) max_new = 512;
  const int total_cap = (think_budget > 0 ? think_budget : 0) + max_new;

  const std::string cfg_path  = model_dir + "/smlx.config.txt";
  const std::string weights   = model_dir + "/model.safetensors";
  const std::string tok_path  = model_dir + "/tokenizer.json";
  const std::string tcfg_path = model_dir + "/tokenizer_config.json";
  const std::string gcfg_path = model_dir + "/generation_config.json";

  /* --- load model + tokenizer once --- */
  smlx_config cfg;
  if (smlx_config_load(&cfg, cfg_path.c_str()) != 0) return 1;
  smlx_model* model = smlx_model_load(&cfg, weights.c_str());
  auto tok = tokenizers::Tokenizer::FromBlobJSON(read_file(tok_path));

  /* --- chat template + stop tokens (built once) --- */
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
    if (tmpl.empty()) { fprintf(stderr, "smlx_chat: no chat_template; use --raw\n"); return 1; }
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
    if (eos.empty()) fprintf(stderr, "smlx_chat: warning, no EOS token found\n");
  }

  smlx_sampling samp{};
  samp.temperature = temp; samp.top_k = top_k; samp.top_p = top_p; samp.seed = seed;

  std::unique_ptr<minja::chat_template> chat;
  if (!raw) chat = std::make_unique<minja::chat_template>(tmpl, bos, eos_tok);

  /* Render history (or raw text) -> ids -> generate, streaming to stdout.
   * Returns the decoded assistant reply. Uses a fresh KV cache per call. */
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

    /* Length of s excluding a trailing run of U+FFFD (EF BF BD), which the
     * tokenizer emits for a multi-byte char that isn't complete yet. Holding
     * those back until more tokens arrive avoids printing replacement boxes. */
    auto valid_end = [](const std::string& s) -> size_t {
      size_t n = s.size();
      while (n >= 3 && (unsigned char)s[n-3] == 0xEF &&
             (unsigned char)s[n-2] == 0xBF && (unsigned char)s[n-1] == 0xBD)
        n -= 3;
      return n;
    };

    /* Tokens for the forced thinking-block close. */
    std::vector<int32_t> close_ids = tok->Encode("\n</think>\n\n");

    smlx_session* sess = smlx_session_new(model);
    std::vector<int32_t> out;
    std::string last;  /* what we've already printed */

    auto emit = [&]() {
      std::string cur = tok->Decode(out);
      size_t ve = valid_end(cur);
      if (ve > last.size()) {
        fwrite(cur.data() + last.size(), 1, ve - last.size(), stdout);
        fflush(stdout);
        last.assign(cur.data(), ve);
      }
    };

    std::vector<int32_t> next_in = ids;  /* first feed = prompt */
    bool forced = false;
    for (int produced = 0; produced < total_cap; produced++) {
      uint32_t next = smlx_generate(sess, next_in.data(), (int)next_in.size(), &samp);
      if (eos.count((int)next)) break;
      out.push_back((int32_t)next);
      emit();
      next_in.assign(1, (int32_t)next);  /* default: feed the token we just sampled */

      /* Thinking budget: if the model has opened <think> but not closed it
       * within the budget, inject </think> so it must produce the answer.
       * No-op for non-thinking output (no <think> ever appears). */
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
    return tok->Decode(out);  /* full reply (untrimmed) for history */
  };

  if (has_prompt) {
    json messages = json::array();
    messages.push_back({{"role", "user"}, {"content", prompt}});
    generate(messages, prompt);
  } else {
    /* Interactive multi-turn chat. */
    fprintf(stderr, "smlx_chat interactive (%s). Ctrl-D or /exit to quit.\n",
            model_dir.c_str());
    json messages = json::array();
    std::string line;
    while (true) {
      fprintf(stderr, "\n> "); fflush(stderr);
      if (!std::getline(std::cin, line)) break;          /* EOF */
      if (line == "/exit" || line == "/quit") break;
      if (line.empty()) continue;
      if (raw) {
        generate(json::array(), line);                   /* no history in raw mode */
        continue;
      }
      messages.push_back({{"role", "user"}, {"content", line}});
      std::string reply = generate(messages, "");
      messages.push_back({{"role", "assistant"}, {"content", reply}});
    }
    fprintf(stderr, "\n");
  }

  smlx_model_free(model);
  return 0;
}
