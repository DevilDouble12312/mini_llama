
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "mini_llama/backend.h"
#include "mini_llama/chat.h"
#include "mini_llama/context.h"
#include "mini_llama/debug.h"
#include "mini_llama/forward.h"
#include "mini_llama/gguf.h"
#include "mini_llama/gguf_loader.h"
#include "mini_llama/gguf_tokenizer.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"
#include "mini_llama/ops.h"
#include "mini_llama/prompt_builder.h"
#include "mini_llama/quant.h"
#include "mini_llama/request_context.h"
#include "mini_llama/sampler.h"
#include "mini_llama/terminal.h"
#include "mini_llama/thread_pool.h"
#include "mini_llama/tokenizer.h"
#include "mini_llama/httplib.h"

namespace mini_llama {

// ---------------------------------------------------------------------------
// Argument parsing helpers
// ---------------------------------------------------------------------------
static bool ParseIntArg(const char* text, int& value) {
  char* end = nullptr;
  int64_t parsed = std::strtoll(text, &end, 10);
  if (end == text || *end != '\0') {
    return false;
  }
  if (parsed < 0 || parsed > 1000000) {
    return false;
  }
  value = static_cast<int>(parsed);
  return true;
}

static bool ParseFloatArg(const char* text, float& value) {
  char* end = nullptr;
  errno = 0;
  double parsed = std::strtod(text, &end);
  if (end == text || *end != '\0') {
    return false;
  }
  if (errno == ERANGE || !std::isfinite(parsed) || parsed < 0.0 ||
      parsed > 10000.0) {
    return false;
  }
  value = static_cast<float>(parsed);
  return true;
}

static bool ParseUintArg(const char* text, unsigned int& value) {
  if (text[0] == '-' || text[0] == '\0') {
    return false;
  }
  char* end = nullptr;
  errno = 0;
  uint64_t parsed = std::strtoull(text, &end, 10);
  if (end == text || *end != '\0') {
    return false;
  }
  if (errno == ERANGE || parsed > std::numeric_limits<unsigned int>::max()) {
    return false;
  }
  value = static_cast<unsigned int>(parsed);
  return true;
}

static bool ParseBackendArg(const char* text, BackendConfig& config) {
  BackendKind kind;
  if (!ParseBackendKind(text, kind)) {
    return false;
  }
  config.kind = kind;
  return true;
}

static bool ParseDeviceArg(const char* text, BackendConfig& config) {
  int device_id = 0;
  if (!ParseIntArg(text, device_id) || device_id < 0) {
    return false;
  }
  config.device_id = device_id;
  config.device_id_set = true;
  return true;
}

static bool ValidateBackendOrPrint(const BackendConfig& config) {
  try {
    ValidateBackend(config);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Backend setup failed: " << e.what() << "\n";
    return false;
  }
}

static void PrintBackendInfo(const BackendConfig& config) {
  std::cout << "backend: " << BackendKindName(config.kind) << "\n";
  std::cout << BackendExecutionNote(config) << "\n";
}

static std::string FormatMb(size_t bytes) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(2)
      << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << " MB";
  return out.str();
}

// ---------------------------------------------------------------------------
// Model-level quantization helpers
// ---------------------------------------------------------------------------
static QuantizedTensor QuantizeLinearToType(const QuantizedTensor& src,
                                            QuantType type) {
  if (src.type != QuantType::kF32) {
    return src;  // Already quantized; leave as-is.
  }
  QuantizedTensor q;
  q.type = type;
  q.shape = src.shape;
  Tensor f32 = ToTensor(src);
  if (type == QuantType::kQ80) {
    q.q8_0_data = QuantizeToQ80(f32);
  } else if (type == QuantType::kQ40) {
    q.q4_0_data = QuantizeToQ40(f32);
  } else {
    throw std::runtime_error("unsupported quantize target type");
  }
  return q;
}

static void QuantizeLayerLinearWeights(LayerWeights& lw, QuantType type) {
  lw.wq = QuantizeLinearToType(lw.wq, type);
  lw.wk = QuantizeLinearToType(lw.wk, type);
  lw.wv = QuantizeLinearToType(lw.wv, type);
  lw.wo = QuantizeLinearToType(lw.wo, type);
  lw.w_gate = QuantizeLinearToType(lw.w_gate, type);
  lw.w_up = QuantizeLinearToType(lw.w_up, type);
  lw.w_down = QuantizeLinearToType(lw.w_down, type);
}

static void QuantizeModelToQ80(MiniLlamaModel& model) {
  model.lm_head = QuantizeLinearToType(model.lm_head, QuantType::kQ80);
  for (auto& lw : model.layers) {
    QuantizeLayerLinearWeights(lw, QuantType::kQ80);
  }
}

static void QuantizeModelToQ40(MiniLlamaModel& model) {
  model.lm_head = QuantizeLinearToType(model.lm_head, QuantType::kQ40);
  for (auto& lw : model.layers) {
    QuantizeLayerLinearWeights(lw, QuantType::kQ40);
  }
}

static void ApplyQuantOverride(MiniLlamaModel& model,
                               const std::string& quant_type) {
  if (quant_type.empty()) {
    return;
  }
  if (quant_type == "q8_0") {
    QuantizeModelToQ80(model);
    return;
  }
  if (quant_type == "q4_0") {
    QuantizeModelToQ40(model);
    return;
  }
  throw std::runtime_error("unsupported quant type: " + quant_type);
}

// ---------------------------------------------------------------------------
// Model weight byte accounting
// ---------------------------------------------------------------------------
static size_t QuantizedTensorBytes(const QuantizedTensor& t) {
  switch (t.type) {
    case QuantType::kF32:
      return t.f32_data.size() * sizeof(float);
    case QuantType::kQ80:
      return t.q8_0_data.size() * sizeof(BlockQ80);
    case QuantType::kQ40:
      return t.q4_0_data.size() * sizeof(BlockQ40);
    case QuantType::kQ41:
      return t.q4_1_data.size() * sizeof(BlockQ41);
  }
  return 0;
}

static size_t ModelWeightBytes(const MiniLlamaModel& model) {
  size_t total = model.token_embedding.size() * sizeof(float);
  total += model.final_norm.size() * sizeof(float);
  total += QuantizedTensorBytes(model.lm_head);
  for (const auto& lw : model.layers) {
    total += lw.attention_norm.size() * sizeof(float);
    total += lw.ffn_norm.size() * sizeof(float);
    total += QuantizedTensorBytes(lw.wq);
    total += QuantizedTensorBytes(lw.wk);
    total += QuantizedTensorBytes(lw.wv);
    total += QuantizedTensorBytes(lw.wo);
    total += QuantizedTensorBytes(lw.w_gate);
    total += QuantizedTensorBytes(lw.w_up);
    total += QuantizedTensorBytes(lw.w_down);
  }
  return total;
}

static size_t ModelWeightBytesF32(const MiniLlamaModel& model) {
  auto f32_bytes = [](const QuantizedTensor& t) {
    return t.num_elements() * sizeof(float);
  };
  size_t total = model.token_embedding.size() * sizeof(float);
  total += model.final_norm.size() * sizeof(float);
  total += f32_bytes(model.lm_head);
  for (const auto& lw : model.layers) {
    total += lw.attention_norm.size() * sizeof(float);
    total += lw.ffn_norm.size() * sizeof(float);
    total += f32_bytes(lw.wq);
    total += f32_bytes(lw.wk);
    total += f32_bytes(lw.wv);
    total += f32_bytes(lw.wo);
    total += f32_bytes(lw.w_gate);
    total += f32_bytes(lw.w_up);
    total += f32_bytes(lw.w_down);
  }
  return total;
}

static Tensor RunLogitsForTokens(const MiniLlamaModel& model,
                                 const std::vector<int>& tokens) {
  MiniLlamaContext ctx(&model);
  MiniBatch batch = MiniBatch::FromTokens(tokens, 0);
  return ForwardBatch(ctx, model, batch);
}

static std::pair<float, float> LogitsError(const Tensor& baseline,
                                           const Tensor& candidate) {
  if (baseline.shape != candidate.shape) {
    throw std::runtime_error(
        "LogitsError: shape mismatch, baseline=" + baseline.ShapeStringShort() +
        ", candidate=" + candidate.ShapeStringShort());
  }
  float max_err = 0.0f;
  float sum_err = 0.0f;
  for (size_t i = 0; i < baseline.size(); ++i) {
    float err = std::abs(baseline.data[i] - candidate.data[i]);
    max_err = std::max(max_err, err);
    sum_err += err;
  }
  return {max_err, sum_err / static_cast<float>(baseline.size())};
}

// ---------------------------------------------------------------------------
// Generate mode
// ---------------------------------------------------------------------------
static void PrintGenerateUsage(const char* prog) {
  std::cout
      << "Usage: " << prog << " generate [options]\n"
      << "Options:\n"
      << "  --model <path|dir>   Path to model weights binary or model "
         "directory (default: models/tiny/model.bin)\n"
      << "  --config <path>      Path to model config JSON (default: "
         "models/tiny/model.json)\n"
      << "  -p, --prompt <str>   Input prompt text (default: \"hello\")\n"
      << "  -n, --n-predict <n>  Number of tokens to generate (default: 16)\n"
      << "  --temperature <T>    Sampling temperature (default: 0.0 = greedy)\n"
      << "  --top-k <k>          Top-k sampling (default: 0 = disabled)\n"
      << "  --seed <S>           Random seed for reproducible sampling "
         "(default: 0 = random)\n"
      << "  --tokenizer <path>   Path to vocab.json tokenizer file\n"
      << "  --quant q8_0|q4_0    Quantize loaded Linear weights before "
         "generation\n"
      << "  --threads <n>        Number of threads for parallel ops (0 = "
         "auto)\n"
      << "  --dump-logits <dir>  Dump logits for each step to directory\n"
      << "  -h, --help           Show this help\n";
}

static void DumpLogits(const Tensor& logits, const std::string& path) {
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("failed to open logits dump file: " + path);
  }
  out.write(reinterpret_cast<const char*>(logits.data.data()),
            static_cast<std::streamsize>(logits.data.size() * sizeof(float)));
  if (!out.good()) {
    throw std::runtime_error("failed to write logits to: " + path);
  }
}

static void EnsureDumpDirectory(const std::string& path) {
  std::error_code error;
  std::filesystem::create_directories(path, error);
  if (error) {
    throw std::runtime_error("failed to create logits dump directory: " + path +
                             ": " + error.message());
  }
  if (!std::filesystem::is_directory(path)) {
    throw std::runtime_error("logits dump path is not a directory: " + path);
  }
}

static void DumpGeneratedTokens(const std::vector<int>& generated,
                                const std::string& path) {
  std::ofstream out(path);
  if (!out.is_open()) {
    throw std::runtime_error("failed to open generation token dump file: " +
                             path);
  }
  for (size_t i = 0; i < generated.size(); ++i) {
    if (i > 0) {
      out << " ";
    }
    out << generated[i];
  }
  out << "\n";
  if (!out.good()) {
    throw std::runtime_error("failed to write generation tokens to: " + path);
  }
}

// ---------------------------------------------------------------------------
// Tokenizer path resolution
// ---------------------------------------------------------------------------

static std::string ResolveManifestTokenizerPath(
    const std::string& config_path) {
  try {
    ModelManifest manifest = ParseManifest(config_path);
    if (manifest.tokenizer.type != "json_vocab" ||
        manifest.tokenizer.path.empty()) {
      return "";
    }
    std::filesystem::path tokenizer_path(manifest.tokenizer.path);
    if (tokenizer_path.is_relative()) {
      std::filesystem::path config_file(config_path);
      tokenizer_path = config_file.parent_path() / tokenizer_path;
    }
    return tokenizer_path.string();
  } catch (const std::exception&) {
    return "";
  }
}

// ---------------------------------------------------------------------------
// Model loading helper: supports both JSON+BIN and GGUF
// ---------------------------------------------------------------------------

struct LoadedModel {
  MiniLlamaModel model;
  std::unique_ptr<ITokenizer> tokenizer;
  std::string chat_template;
};

static bool IsGgufFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    return false;
  }
  char magic[4];
  return f.read(magic, 4) && std::memcmp(magic, "GGUF", 4) == 0;
}

static std::string FindGgufInDirectory(const std::string& path) {
  std::vector<std::filesystem::path> candidates;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(path, ec)) {
    if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
      candidates.push_back(entry.path());
    }
  }
  if (candidates.empty()) {
    return "";
  }
  std::sort(candidates.begin(), candidates.end());
  return candidates.front().string();
}

static std::unique_ptr<ITokenizer> CreateTokenizerFromVocabHint(
    const std::string& vocab_path) {
  std::filesystem::path vocab(vocab_path);
  std::filesystem::path dir = vocab.parent_path();
  std::filesystem::path merges = dir / "merges.txt";
  std::filesystem::path special = dir / "special_tokens.json";
  if (std::filesystem::exists(merges)) {
    return CreateBpeTokenizer(vocab.string(), merges.string(),
                              special.string());
  }
  return CreateTokenizer(vocab.string());
}

static LoadedModel LoadModelAndTokenizer(
    const std::string& path, const std::string& explicit_config_path = "",
    const std::string& explicit_tokenizer_path = "") {
  LoadedModel result;

  std::string model_path = path;
  std::string config_path;
  std::string tokenizer_path;
  if (std::filesystem::is_directory(path)) {
    std::filesystem::path bin_path = std::filesystem::path(path) / "model.bin";
    std::filesystem::path json_path =
        std::filesystem::path(path) / "model.json";
    std::string gguf_path = FindGgufInDirectory(path);
    if ((!std::filesystem::exists(bin_path) ||
         !std::filesystem::exists(json_path)) &&
        !gguf_path.empty()) {
      model_path = gguf_path;
    }
  }

  bool is_gguf = IsGgufFile(model_path);

  if (is_gguf) {
    result.model = LoadGgufModel(model_path);
    if (!result.model.loaded) {
      return result;
    }

    // 1. Try to Load tokenizer from GGUF metadata (M14)
    if (!explicit_tokenizer_path.empty()) {
      result.tokenizer = CreateTokenizerFromVocabHint(explicit_tokenizer_path);
    } else {
      result.tokenizer = CreateGgufTokenizer(model_path);
    }

    // 2. Fallback to external vocab.json + merges.txt if GGUF has no tokenizer
    // metadata
    if (!result.tokenizer) {
      std::filesystem::path gguf_dir =
          std::filesystem::path(model_path).parent_path();
      std::string vocab_path = (gguf_dir / "vocab.json").string();
      std::string merges_path = (gguf_dir / "merges.txt").string();
      std::string special_path = (gguf_dir / "special_tokens.json").string();
      if (std::filesystem::exists(vocab_path) &&
          std::filesystem::exists(merges_path)) {
        result.tokenizer =
            CreateBpeTokenizer(vocab_path, merges_path, special_path);
      }
    }

    // 3. Load chat template from GGUF metadata (M14)
    result.chat_template = LoadChatTemplateFromGguf(model_path);
  } else {
    // Directory-based JSON+BIN format
    if (std::filesystem::is_directory(path)) {
      model_path = path + "/model.bin";
      config_path = path + "/model.json";
    } else {
      model_path = path;
      if (!explicit_config_path.empty()) {
        config_path = explicit_config_path;
      } else {
        config_path =
            std::filesystem::path(path).parent_path().string() + "/model.json";
      }
    }
    result.model = LoadModel(config_path, model_path);
    if (!result.model.loaded) {
      return result;
    }
    if (!explicit_tokenizer_path.empty()) {
      tokenizer_path = explicit_tokenizer_path;
    }
    if (tokenizer_path.empty()) {
      std::filesystem::path dir =
          std::filesystem::is_directory(path)
              ? std::filesystem::path(path)
              : std::filesystem::path(path).parent_path();
      std::filesystem::path auto_vocab = dir / "vocab.json";
      if (std::filesystem::exists(auto_vocab)) {
        tokenizer_path = auto_vocab.string();
      }
    }
    if (tokenizer_path.empty()) {
      tokenizer_path = ResolveManifestTokenizerPath(config_path);
    }
    result.tokenizer = CreateTokenizer(tokenizer_path);
  }

  if (!result.tokenizer) {
    result.model.load_error = "Failed to load tokenizer";
    result.model.loaded = false;
    return result;
  }

  return result;
}

static size_t CommonPrefixLength(const std::vector<int>& a,
                                 const std::vector<int>& b) {
  size_t n = std::min(a.size(), b.size());
  size_t i = 0;
  while (i < n && a[i] == b[i]) {
    ++i;
  }
  return i;
}

static void PrintRequestTrace(const RequestContext& request,
                              std::ostream& out = std::cout) {
  for (const std::string& line : FormatRequestTraceEvents(request)) {
    out << line << "\n";
  }
  out << FormatRequestTraceSummary(request) << "\n";
}

static int RunGenerate(int argc, char** argv) {
  std::string model_path = "models/tiny/model.bin";
  std::string config_path = "models/tiny/model.json";
  bool config_path_set = false;
  std::string prompt = "hello";
  int n_predict = 16;
  float temperature = 0.0f;
  int top_k = 0;
  unsigned int seed = 0;
  std::string dump_logits_dir;
  std::string tokenizer_path;
  std::string quant_type;
  int n_threads = 0;
  BackendConfig backend_config;

  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
      model_path = argv[++i];
    } else if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
      config_path = argv[++i];
      config_path_set = true;
    } else if ((std::strcmp(argv[i], "-p") == 0 ||
                std::strcmp(argv[i], "--prompt") == 0) &&
               i + 1 < argc) {
      prompt = argv[++i];
    } else if ((std::strcmp(argv[i], "-n") == 0 ||
                std::strcmp(argv[i], "--n-predict") == 0) &&
               i + 1 < argc) {
      if (!ParseIntArg(argv[++i], n_predict)) {
        std::cerr
            << "Invalid --n-predict value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
      if (!ParseFloatArg(argv[++i], temperature)) {
        std::cerr
            << "Invalid --temperature value. Expected a non-negative float.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
      if (!ParseIntArg(argv[++i], top_k)) {
        std::cerr
            << "Invalid --top-k value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      if (!ParseUintArg(argv[++i], seed)) {
        std::cerr << "Invalid --seed value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--dump-logits") == 0 && i + 1 < argc) {
      dump_logits_dir = argv[++i];
    } else if (std::strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
      tokenizer_path = argv[++i];
    } else if (std::strcmp(argv[i], "--quant") == 0 && i + 1 < argc) {
      quant_type = argv[++i];
    } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      if (!ParseIntArg(argv[++i], n_threads) || n_threads < 0) {
        std::cerr << "Invalid --threads value.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
      if (!ParseBackendArg(argv[++i], backend_config)) {
        std::cerr << "Invalid --backend value. Supported values: cpu.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
      if (!ParseDeviceArg(argv[++i], backend_config)) {
        std::cerr
            << "Invalid --device value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "-h") == 0 ||
               std::strcmp(argv[i], "--help") == 0) {
      PrintGenerateUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown or incomplete argument: " << argv[i] << "\n";
      PrintGenerateUsage(argv[0]);
      return 1;
    }
  }

  if (!quant_type.empty() && quant_type != "q8_0" && quant_type != "q4_0") {
    std::cerr << "Invalid --quant value: " << quant_type
              << ". Supported values: q8_0, q4_0.\n";
    return 1;
  }

  if (!ValidateBackendOrPrint(backend_config)) {
    return 1;
  }

  std::cout << "mini-llama.cpp\n";
  std::cout << "==============\n\n";
  PrintBackendInfo(backend_config);

  if (!dump_logits_dir.empty()) {
    try {
      EnsureDumpDirectory(dump_logits_dir);
    } catch (const std::exception& e) {
      std::cerr << "Logits dump setup failed: " << e.what() << "\n";
      return 1;
    }
  }

  // For backward compat, if --config was explicitly set but --model is the
  // default, override model_path to be the config's directory.
  if (config_path_set && model_path == "models/tiny/model.bin") {
    model_path = std::filesystem::path(config_path).parent_path().string() +
                 "/model.bin";
  }

  RequestContext request =
      StartRequest("generate", BackendKindName(backend_config.kind), model_path);

  auto stage_start = RequestClock::now();
  LoadedModel lm = LoadModelAndTokenizer(
      model_path, config_path_set ? config_path : "", tokenizer_path);
  request.model_load_ms = ElapsedMs(stage_start);
  request.RecordEvent("model_load", request.model_load_ms, 0, model_path);
  if (!lm.model.loaded) {
    request.SetError("Failed to load model: " + lm.model.load_error);
    request.Finish();
    PrintRequestTrace(request, std::cerr);
    std::cerr << request.error << "\n";
    return 1;
  }
  if (!lm.tokenizer) {
    request.SetError("Failed to load tokenizer.");
    request.Finish();
    PrintRequestTrace(request, std::cerr);
    std::cerr << request.error << "\n";
    return 1;
  }
  if (lm.model.config.vocab_size < lm.tokenizer->vocab_size()) {
    request.SetError("Model vocab_size must be at least " +
                     std::to_string(lm.tokenizer->vocab_size()) +
                     " for the tokenizer.");
    request.Finish();
    PrintRequestTrace(request, std::cerr);
    std::cerr << request.error << "\n";
    return 1;
  }
  MiniLlamaModel& model = lm.model;
  try {
    stage_start = RequestClock::now();
    ApplyQuantOverride(model, quant_type);
    request.RecordEvent("quantize", ElapsedMs(stage_start), 0,
                        quant_type.empty() ? "model-native" : quant_type);
  } catch (const std::exception& e) {
    request.SetError("Quantization failed: " + std::string(e.what()));
    request.Finish();
    PrintRequestTrace(request, std::cerr);
    std::cerr << request.error << "\n";
    return 1;
  }
  std::unique_ptr<ITokenizer>& tokenizer = lm.tokenizer;
  stage_start = RequestClock::now();
  std::vector<int> tokens = tokenizer->Encode(prompt);
  request.tokenize_ms = ElapsedMs(stage_start);
  request.prompt_tokens = static_cast<int>(tokens.size());
  request.RecordEvent("tokenize", request.tokenize_ms, request.prompt_tokens,
                      "prompt");
  std::cout << "prompt: " << prompt << "\n";
  std::cout << "tokens: [";
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << tokens[i];
  }
  std::cout << "]\n";
  SetThreadCount(n_threads);
  std::cout << "sampling: temperature=" << temperature << ", top_k=" << top_k
            << ", seed=" << seed << "\n";
  std::cout << "quant: " << (quant_type.empty() ? "model-native" : quant_type)
            << "\n";
  std::cout << "threads: " << GetThreadCount() << "\n\n";

  if (tokens.size() > static_cast<size_t>(model.config.max_seq_len)) {
    request.SetError("Prompt is too long for max_seq_len=" +
                     std::to_string(model.config.max_seq_len) + ".");
    request.Finish();
    PrintRequestTrace(request, std::cerr);
    std::cerr << request.error << "\n";
    return 1;
  }
  if (tokens.size() + static_cast<size_t>(n_predict) >
      static_cast<size_t>(model.config.max_seq_len)) {
    request.SetError("Requested tokens exceed context window.");
    request.Finish();
    PrintRequestTrace(request, std::cerr);
    std::cerr << request.error << "\n";
    return 1;
  }

  MiniLlamaContext ctx(&model);
  SamplingParams sampling_params;
  sampling_params.temperature = temperature;
  sampling_params.top_k = top_k;
  sampling_params.seed = seed;
  MiniSampler sampler(sampling_params);

  size_t prompt_len = tokens.size();
  int step = 0;
  try {
    Tensor logits;
    std::cout << "prefill...\n";
    stage_start = RequestClock::now();
    if (!dump_logits_dir.empty()) {
      for (size_t i = 0; i < prompt_len; ++i) {
        MiniBatch prefill_step =
            MiniBatch::FromTokens({tokens[i]}, static_cast<int>(i));
        logits = ForwardBatch(ctx, model, prefill_step);
        ++ctx.n_prefill_tokens;
        DumpLogits(logits, dump_logits_dir + "/logits_step" +
                               std::to_string(step) + ".bin");
        ++step;
      }
    } else {
      MiniBatch prefill = MiniBatch::FromTokens(tokens, 0);
      logits = ForwardBatch(ctx, model, prefill);
      ctx.n_prefill_tokens += static_cast<int>(tokens.size());
    }
    request.prefill_ms = ElapsedMs(stage_start);
    request.prefill_tokens = static_cast<int>(prompt_len);
    request.RecordEvent("prefill", request.prefill_ms, request.prefill_tokens,
                        dump_logits_dir.empty() ? "batch" : "step_dump");

    if (n_predict > 0) {
      stage_start = RequestClock::now();
      int next_token = sampler.Sample(logits, sampling_params);
      request.sample_ms += ElapsedMs(stage_start);
      tokens.push_back(next_token);

      std::cout << "decode loop...\n";
      for (int i = 1; i < n_predict; ++i) {
        MiniBatch decode_batch = MiniBatch::Single(
            tokens.back(), static_cast<int>(tokens.size() - 1));
        stage_start = RequestClock::now();
        logits = ForwardBatch(ctx, model, decode_batch);
        double decode_ms = ElapsedMs(stage_start);
        request.decode_ms += decode_ms;
        request.RecordEvent("decode", decode_ms, 1,
                            "pos=" + std::to_string(tokens.size() - 1));
        ++ctx.n_decode_tokens;
        ++request.decode_tokens;
        if (!dump_logits_dir.empty()) {
          DumpLogits(logits, dump_logits_dir + "/logits_step" +
                                 std::to_string(step) + ".bin");
          ++step;
        }
        stage_start = RequestClock::now();
        next_token = sampler.Sample(logits, sampling_params);
        request.sample_ms += ElapsedMs(stage_start);
        tokens.push_back(next_token);
        if (next_token == tokenizer->eos_id()) {
          break;
        }
      }
    } else {
      std::cout << "decode loop skipped.\n";
    }
  } catch (const std::exception& e) {
    request.SetError("Inference failed: " + std::string(e.what()));
    request.Finish();
    PrintRequestTrace(request, std::cerr);
    std::cerr << request.error << "\n";
    return 1;
  }

  if (!dump_logits_dir.empty()) {
    std::vector<int> generated(tokens.begin() + prompt_len, tokens.end());
    try {
      DumpGeneratedTokens(generated,
                          dump_logits_dir + "/generation_tokens.txt");
    } catch (const std::exception& e) {
      request.SetError("Logits dump failed: " + std::string(e.what()));
      request.Finish();
      PrintRequestTrace(request, std::cerr);
      std::cerr << request.error << "\n";
      return 1;
    }
  }

  std::vector<int> generated(tokens.begin() + prompt_len, tokens.end());
  request.generated_tokens = static_cast<int>(generated.size());
  request.RecordEvent("sample", request.sample_ms, request.generated_tokens,
                      "generated_tokens");
  request.Finish();
  std::cout << "\ngenerated tokens: [";
  for (size_t i = 0; i < generated.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << generated[i];
  }
  std::cout << "]\n";

  std::string generated_text = tokenizer->Decode(generated);
  std::cout << "generated text: \"" << generated_text << "\"\n";
  PrintRequestTrace(request);

  return 0;
}

// ---------------------------------------------------------------------------
// Run mode (interactive chat)
// ---------------------------------------------------------------------------
static void PrintRunUsage(const char* prog) {
  std::cout
      << "Usage: " << prog << " run <model-path|dir> [options]\n"
      << "Options:\n"
      << "  --temperature <T>  Sampling temperature (default: 0.0 = greedy)\n"
      << "  --top-k <k>        Top-k sampling (default: 0 = disabled)\n"
      << "  --seed <S>         Random seed (default: 0 = random)\n"
      << "  -n, --n-predict <n> Maximum response tokens per turn (default: "
         "64)\n"
      << "  --tokenizer <path> Path to vocab.json tokenizer file\n"
      << "  -h, --help         Show this help\n";
}

static int RunChat(int argc, char** argv) {
  if (argc >= 3 && (std::strcmp(argv[2], "-h") == 0 ||
                    std::strcmp(argv[2], "--help") == 0)) {
    PrintRunUsage(argv[0]);
    return 0;
  }

  if (argc < 3) {
    std::cerr << "Missing model directory.\n";
    PrintRunUsage(argv[0]);
    return 1;
  }

  std::string model_dir = argv[2];
  float temperature = 0.0f;
  int top_k = 0;
  unsigned int seed = 0;
  int max_response_tokens = 64;
  std::string tokenizer_path;
  BackendConfig backend_config;

  for (int i = 3; i < argc; ++i) {
    if (std::strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
      if (!ParseFloatArg(argv[++i], temperature)) {
        std::cerr
            << "Invalid --temperature value. Expected a non-negative float.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
      if (!ParseIntArg(argv[++i], top_k)) {
        std::cerr
            << "Invalid --top-k value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      if (!ParseUintArg(argv[++i], seed)) {
        std::cerr << "Invalid --seed value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if ((std::strcmp(argv[i], "-n") == 0 ||
                std::strcmp(argv[i], "--n-predict") == 0) &&
               i + 1 < argc) {
      if (!ParseIntArg(argv[++i], max_response_tokens) ||
          max_response_tokens <= 0) {
        std::cerr
            << "Invalid --n-predict value. Expected a positive integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
      tokenizer_path = argv[++i];
    } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
      if (!ParseBackendArg(argv[++i], backend_config)) {
        std::cerr << "Invalid --backend value. Supported values: cpu.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
      if (!ParseDeviceArg(argv[++i], backend_config)) {
        std::cerr
            << "Invalid --device value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "-h") == 0 ||
               std::strcmp(argv[i], "--help") == 0) {
      PrintRunUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << argv[i] << "\n";
      PrintRunUsage(argv[0]);
      return 1;
    }
  }

  if (!ValidateBackendOrPrint(backend_config)) {
    return 1;
  }

  // Load model
  LoadedModel lm = LoadModelAndTokenizer(model_dir, "", tokenizer_path);
  if (!lm.model.loaded) {
    std::cerr << "Failed to load model: " << lm.model.load_error << "\n";
    return 1;
  }
  if (!lm.tokenizer) {
    std::cerr << "Failed to load tokenizer.\n";
    return 1;
  }
  if (lm.model.config.vocab_size < lm.tokenizer->vocab_size()) {
    std::cerr << "Model vocab_size must be at least "
              << lm.tokenizer->vocab_size() << " for the tokenizer.\n";
    return 1;
  }

  MiniLlamaModel& model = lm.model;
  std::unique_ptr<ITokenizer>& tokenizer = lm.tokenizer;

  PromptBuilder builder;
  if (!lm.chat_template.empty()) {
    builder.SetChatTemplate(lm.chat_template);
  }
  Terminal term;
  ChatSession session;
  session.sampling_params.temperature = temperature;
  session.sampling_params.top_k = top_k;
  session.sampling_params.seed = seed;

  // Plain text mode keeps the default system message in session state.
  if (lm.chat_template.empty()) {
    session.AddMessage("system", "You are a helpful assistant.");
  }

  term.PrintMessage("mini-llama.cpp chat");
  term.PrintMessage("backend: " + BackendKindName(backend_config.kind));
  term.PrintMessage(BackendExecutionNote(backend_config));
  term.PrintMessage("Type /help for commands, /exit to quit.\n");
  if (lm.chat_template.empty() && model.config.max_seq_len <= 256) {
    term.PrintMessage(
        "Tiny teaching model: random weights, small context window, smoke-test "
        "output.");
    term.PrintMessage(
        "Real chat demo: ./build/mini-llama run models/chat -n 8\n");
  }

  MiniLlamaContext ctx(&model);

  while (true) {
    term.PrintUserPrompt();
    std::string input = term.ReadLine();
    if (input.empty() && std::cin.eof()) {
      break;
    }

    // Handle commands
    if (input == "/help") {
      term.PrintHelp();
      continue;
    }
    if (input == "/exit") {
      break;
    }
    if (input == "/clear") {
      session.Clear();
      ctx = MiniLlamaContext(&model);
      if (lm.chat_template.empty()) {
        session.AddMessage("system", "You are a helpful assistant.");
      }
      term.PrintMessage("Chat history cleared.\n");
      continue;
    }
    if (input == "/stats") {
      term.PrintStats(session);
      continue;
    }
    if (input == "/params") {
      term.PrintParams(session.sampling_params);
      continue;
    }
    if (!input.empty() && input[0] == '/') {
      term.PrintMessage("Unknown command: " + input + "\n");
      continue;
    }
    if (input.empty()) {
      continue;
    }

    RequestContext request =
        StartRequest("run", BackendKindName(backend_config.kind), model_dir);

    std::vector<ChatMessage> candidate_messages = session.messages;
    candidate_messages.push_back({"user", input});

    auto stage_start = RequestClock::now();
    std::string prompt_text = builder.Build(candidate_messages);
    request.RecordEvent("prompt_build", ElapsedMs(stage_start), 0,
                        "messages=" +
                            std::to_string(candidate_messages.size()));
    stage_start = RequestClock::now();
    std::vector<int> tokens = tokenizer->Encode(prompt_text);
    request.tokenize_ms = ElapsedMs(stage_start);
    request.prompt_tokens = static_cast<int>(tokens.size());
    request.RecordEvent("tokenize", request.tokenize_ms, request.prompt_tokens,
                        "prompt");

    if (tokens.size() >= static_cast<size_t>(model.config.max_seq_len)) {
      request.SetError("prompt uses " + std::to_string(tokens.size()) +
                       " tokens, context window is " +
                       std::to_string(model.config.max_seq_len) +
                       ". Use /clear or a shorter prompt.");
      request.Finish();
      PrintRequestTrace(request);
      term.PrintMessage("Error: " + request.error + "\n");
      continue;
    }

    int max_response =
        model.config.max_seq_len - static_cast<int>(tokens.size());
    if (max_response > max_response_tokens) {
      max_response = max_response_tokens;
    }

    session.messages = candidate_messages;
    MiniSampler sampler(session.sampling_params);
    Tensor logits;

    auto start = std::chrono::steady_clock::now();

    size_t cached_prefix_len = session.LongestCachedPrefix(tokens);
    size_t context_prefix_len = CommonPrefixLength(ctx.token_history, tokens);
    size_t prefix_len = std::min(cached_prefix_len, context_prefix_len);
    if (prefix_len >= tokens.size()) {
      ctx = MiniLlamaContext(&model);
      prefix_len = 0;
    } else if (prefix_len == 0) {
      ctx = MiniLlamaContext(&model);
    } else if (prefix_len < ctx.token_history.size()) {
      ctx.token_history.resize(prefix_len);
      ctx.pos = static_cast<int>(prefix_len - 1);
    }
    std::vector<int> new_prompt_tokens(
        tokens.begin() + static_cast<std::ptrdiff_t>(prefix_len), tokens.end());
    session.SetTokenHistory(tokens);

    // Prefill only the context suffix that is missing from KV cache.
    {
      MiniBatch prefill = MiniBatch::FromTokens(new_prompt_tokens,
                                                static_cast<int>(prefix_len));
      stage_start = RequestClock::now();
      logits = ForwardBatch(ctx, model, prefill);
      request.prefill_ms = ElapsedMs(stage_start);
      request.prefill_tokens = static_cast<int>(new_prompt_tokens.size());
      request.RecordEvent(
          "prefill", request.prefill_ms, request.prefill_tokens,
          "radix_hit=" + std::to_string(cached_prefix_len) +
              ", prefix_reuse=" + std::to_string(prefix_len));
      ctx.n_prefill_tokens += static_cast<int>(new_prompt_tokens.size());
    }

    // Generate response
    std::vector<int> generated_ids;
    std::string streamed_reply;
    int generated_count = 0;

    term.PrintAssistantPrefix();
    try {
      for (int i = 0; i < max_response; ++i) {
        stage_start = RequestClock::now();
        int next_token = sampler.Sample(logits, session.sampling_params);
        request.sample_ms += ElapsedMs(stage_start);
        tokens.push_back(next_token);
        session.AppendToken(next_token);
        ++generated_count;

        if (next_token != tokenizer->eos_id()) {
          generated_ids.push_back(next_token);
          std::string current_reply = tokenizer->Decode(generated_ids);
          if (current_reply.size() > streamed_reply.size()) {
            term.PrintTokenText(current_reply.substr(streamed_reply.size()));
            term.Flush();
            streamed_reply = current_reply;
          }
        }

        MiniBatch decode_batch =
            MiniBatch::Single(next_token, static_cast<int>(tokens.size() - 1));
        stage_start = RequestClock::now();
        logits = ForwardBatch(ctx, model, decode_batch);
        double decode_ms = ElapsedMs(stage_start);
        request.decode_ms += decode_ms;
        request.RecordEvent("decode", decode_ms, 1,
                            "pos=" + std::to_string(tokens.size() - 1));
        ++ctx.n_decode_tokens;
        ++request.decode_tokens;
        if (next_token == tokenizer->eos_id()) {
          break;
        }
      }
    } catch (const std::exception& e) {
      term.NewLine();
      request.SetError("Inference error: " + std::string(e.what()));
      request.Finish();
      PrintRequestTrace(request);
      term.PrintMessage(request.error + "\n");
      continue;
    }

    // Decode and output
    std::string assistant_reply = tokenizer->Decode(generated_ids);
    if (assistant_reply.size() > streamed_reply.size()) {
      term.PrintTokenText(assistant_reply.substr(streamed_reply.size()));
    }
    term.NewLine();
    term.NewLine();

    auto end = std::chrono::steady_clock::now();
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    session.AddMessage("assistant", assistant_reply);
    session.RecordTurn(static_cast<int>(new_prompt_tokens.size()),
                       generated_count, elapsed_ms);
    request.generated_tokens = generated_count;
    request.RecordEvent("sample", request.sample_ms, request.generated_tokens,
                        "generated_tokens");
    session.RecordPrefix(session.token_history);
    request.Finish();
    PrintRequestTrace(request);
  }

  term.PrintMessage("Goodbye.\n");
  return 0;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
static void PrintGlobalUsage(const char* prog) {
  std::cout << "Usage: " << prog << " <command> [options]\n"
            << "Commands:\n"
            << "  generate      One-shot text generation (default)\n"
            << "  run           Interactive chat mode\n"
            << "  inspect       Inspect model metadata\n"
            << "  inspect-gguf  Inspect GGUF file\n"
            << "  bench         Benchmark inference performance\n"
            << "\n"
            << "Run \"" << prog << " generate --help\", \"" << prog
            << " run --help\", or \"" << prog
            << " bench --help\" for details.\n";
}

// ---------------------------------------------------------------------------
// Inspect mode
// ---------------------------------------------------------------------------
static void PrintInspectUsage(const char* prog) {
  std::cout << "Usage: " << prog << " inspect <model-path|dir> [options]\n"
            << "Options:\n"
            << "  -h, --help           Show this help\n";
}

static MiniLlamaModel LoadModelForInspect(const std::string& path) {
  std::string model_path = path;
  if (std::filesystem::is_directory(path)) {
    std::filesystem::path bin_path = std::filesystem::path(path) / "model.bin";
    std::filesystem::path json_path =
        std::filesystem::path(path) / "model.json";
    if (std::filesystem::exists(bin_path) &&
        std::filesystem::exists(json_path)) {
      return LoadModel(json_path.string(), bin_path.string());
    }
    std::string gguf_path = FindGgufInDirectory(path);
    if (!gguf_path.empty()) {
      model_path = gguf_path;
    } else {
      return LoadModel(json_path.string(), bin_path.string());
    }
  }

  if (IsGgufFile(model_path)) {
    return LoadGgufModel(model_path);
  }

  std::filesystem::path model_file(model_path);
  std::string config_path = (model_file.parent_path() / "model.json").string();
  return LoadModel(config_path, model_path);
}

static int RunInspect(int argc, char** argv) {
  if (argc >= 3 && (std::strcmp(argv[2], "-h") == 0 ||
                    std::strcmp(argv[2], "--help") == 0)) {
    PrintInspectUsage(argv[0]);
    return 0;
  }

  if (argc < 3) {
    std::cerr << "Missing model path.\n";
    PrintInspectUsage(argv[0]);
    return 1;
  }

  std::string model_path = argv[2];
  BackendConfig backend_config;
  for (int i = 3; i < argc; ++i) {
    if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
      if (!ParseBackendArg(argv[++i], backend_config)) {
        std::cerr << "Invalid --backend value. Supported values: cpu.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
      if (!ParseDeviceArg(argv[++i], backend_config)) {
        std::cerr
            << "Invalid --device value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "-h") == 0 ||
               std::strcmp(argv[i], "--help") == 0) {
      PrintInspectUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << argv[i] << "\n";
      PrintInspectUsage(argv[0]);
      return 1;
    }
  }

  if (!ValidateBackendOrPrint(backend_config)) {
    return 1;
  }

  if (std::filesystem::is_directory(model_path)) {
    std::filesystem::path config_path =
        std::filesystem::path(model_path) / "model.json";
    if (std::filesystem::exists(config_path) &&
        !mini_llama::InspectModel(config_path.string())) {
      return 1;
    }
  }

  MiniLlamaModel model = LoadModelForInspect(model_path);
  if (!model.loaded) {
    std::cerr << "Failed to load model: " << model.load_error << "\n";
    return 1;
  }

  std::cout << "backend: " << BackendKindName(backend_config.kind) << "\n";

  return 0;
}

static int RunInspectGguf(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " inspect-gguf <path>\n";
    return 1;
  }
  GgufReader reader;
  if (!reader.Load(argv[2])) {
    std::cerr << "Failed to load GGUF: " << reader.load_error << "\n";
    return 1;
  }
  InspectGguf(reader);
  return 0;
}

// ---------------------------------------------------------------------------
// Bench mode
// ---------------------------------------------------------------------------
static void PrintBenchUsage(const char* prog) {
  std::cout
      << "Usage: " << prog << " bench <model-path|dir> [options]\n"
      << "Options:\n"
      << "  -p, --prompt <str>    Input prompt text (default: \"hello\")\n"
      << "  -n, --n-predict <n>   Number of tokens to generate (default: 64)\n"
      << "  --seed <S>            Random seed (default: 0 = random)\n"
      << "  --tokenizer <path>    Path to vocab.json tokenizer file\n"
      << "  --quant q8_0|q4_0     Quantize loaded Linear weights before "
         "benchmark\n"
      << "  --threads <n>         Number of threads for parallel ops (0 = "
         "auto)\n"
      << "  --verbose             Print debug dumps after each step\n"
      << "  -h, --help            Show this help\n";
}

static double BenchmarkRepeated(int warmup, int iterations,
                                const std::function<void()>& fn) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    fn();
  }
  auto end = std::chrono::steady_clock::now();
  double elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  return elapsed_ms / static_cast<double>(iterations);
}

static int RunBench(int argc, char** argv) {
  if (argc >= 3 && (std::strcmp(argv[2], "-h") == 0 ||
                    std::strcmp(argv[2], "--help") == 0)) {
    PrintBenchUsage(argv[0]);
    return 0;
  }

  if (argc < 3) {
    std::cerr << "Missing model directory.\n";
    PrintBenchUsage(argv[0]);
    return 1;
  }

  std::string model_dir = argv[2];
  std::string prompt = "hello";
  int n_predict = 64;
  unsigned int seed = 0;
  std::string tokenizer_path;
  bool verbose = false;
  std::string quant_type;
  int n_threads = 0;
  BackendConfig backend_config;

  for (int i = 3; i < argc; ++i) {
    if ((std::strcmp(argv[i], "-p") == 0 ||
         std::strcmp(argv[i], "--prompt") == 0) &&
        i + 1 < argc) {
      prompt = argv[++i];
    } else if ((std::strcmp(argv[i], "-n") == 0 ||
                std::strcmp(argv[i], "--n-predict") == 0) &&
               i + 1 < argc) {
      if (!ParseIntArg(argv[++i], n_predict)) {
        std::cerr << "Invalid --n-predict value.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      if (!ParseUintArg(argv[++i], seed)) {
        std::cerr << "Invalid --seed value.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
      tokenizer_path = argv[++i];
    } else if (std::strcmp(argv[i], "--quant") == 0 && i + 1 < argc) {
      quant_type = argv[++i];
    } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      if (!ParseIntArg(argv[++i], n_threads) || n_threads < 0) {
        std::cerr << "Invalid --threads value.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
      if (!ParseBackendArg(argv[++i], backend_config)) {
        std::cerr << "Invalid --backend value. Supported values: cpu.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
      if (!ParseDeviceArg(argv[++i], backend_config)) {
        std::cerr
            << "Invalid --device value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--verbose") == 0) {
      verbose = true;
    } else if (std::strcmp(argv[i], "-h") == 0 ||
               std::strcmp(argv[i], "--help") == 0) {
      PrintBenchUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << argv[i] << "\n";
      PrintBenchUsage(argv[0]);
      return 1;
    }
  }

  if (!quant_type.empty() && quant_type != "q8_0" && quant_type != "q4_0") {
    std::cerr << "Invalid --quant value: " << quant_type
              << ". Supported values: q8_0, q4_0.\n";
    return 1;
  }

  if (!ValidateBackendOrPrint(backend_config)) {
    return 1;
  }

  LoadedModel lm = LoadModelAndTokenizer(model_dir, "", tokenizer_path);
  if (!lm.model.loaded) {
    std::cerr << "Failed to load model: " << lm.model.load_error << "\n";
    return 1;
  }
  if (!lm.tokenizer) {
    std::cerr << "Failed to load tokenizer.\n";
    return 1;
  }
  if (lm.model.config.vocab_size < lm.tokenizer->vocab_size()) {
    std::cerr << "Model vocab_size must be at least "
              << lm.tokenizer->vocab_size() << " for the tokenizer.\n";
    return 1;
  }

  MiniLlamaModel baseline_model = lm.model;
  MiniLlamaModel& model = lm.model;
  std::unique_ptr<ITokenizer>& tokenizer = lm.tokenizer;
  std::vector<int> tokens = tokenizer->Encode(prompt);
  if (tokens.size() > static_cast<size_t>(model.config.max_seq_len)) {
    std::cerr << "Prompt too long.\n";
    return 1;
  }
  if (tokens.size() + static_cast<size_t>(n_predict) >
      static_cast<size_t>(model.config.max_seq_len)) {
    n_predict = model.config.max_seq_len - static_cast<int>(tokens.size());
  }

  try {
    ApplyQuantOverride(model, quant_type);
  } catch (const std::exception& e) {
    std::cerr << "Quantization failed: " << e.what() << "\n";
    return 1;
  }
  std::cout << "Benchmark: " << model_dir << "\n";
  std::cout << "  backend: " << BackendKindName(backend_config.kind) << "\n";
  std::cout << "  " << BackendExecutionNote(backend_config) << "\n";
  std::cout << "  prompt: \"" << prompt << "\" (" << tokens.size()
            << " tokens)\n";
  SetThreadCount(n_threads);
  std::cout << "  n_predict: " << n_predict << "\n";
  std::cout << "  seed: " << seed << "\n";
  std::cout << "  quant: " << (quant_type.empty() ? "model-native" : quant_type)
            << "\n";
  std::cout << "  threads: " << GetThreadCount() << "\n";
  std::cout << "  verbose: " << (verbose ? "true" : "false") << "\n\n";

  BenchmarkResult result =
      RunBenchmark(model, tokens, n_predict, seed, verbose);

  std::cout << "Results:\n";
  std::cout << "  prompt tokens:     " << result.n_prompt_tokens << "\n";
  std::cout << "  generated tokens:  " << result.n_generated_tokens << "\n";
  std::cout << "  Decode tokens:     " << result.n_decode_tokens << "\n";
  std::cout << "  prefill time:      " << std::fixed << std::setprecision(2)
            << result.prefill_ms << " ms\n";
  std::cout << "  Decode time:       " << std::fixed << std::setprecision(2)
            << result.decode_ms << " ms\n";
  std::cout << "  total time:        " << std::fixed << std::setprecision(2)
            << (result.prefill_ms + result.decode_ms) << " ms\n";
  std::cout << "  tokens/s (total):  " << std::fixed << std::setprecision(2)
            << result.tokens_per_sec() << "\n";
  std::cout << "  tokens/s (Decode): " << std::fixed << std::setprecision(2)
            << result.decode_tokens_per_sec() << "\n";

  // Memory footprint
  size_t actual_bytes = ModelWeightBytes(model);
  size_t f32_bytes = ModelWeightBytesF32(model);
  std::cout << "\n  weight memory:\n";
  std::cout << "    actual:    " << actual_bytes << " bytes (" << std::fixed
            << std::setprecision(2) << (actual_bytes / (1024.0 * 1024.0))
            << " MB)\n";
  std::cout << "    f32 equiv: " << f32_bytes << " bytes (" << std::fixed
            << std::setprecision(2) << (f32_bytes / (1024.0 * 1024.0))
            << " MB)\n";
  std::cout << "    savings:   " << std::fixed << std::setprecision(2)
            << (static_cast<double>(f32_bytes) / actual_bytes)
            << "x compression\n";

  if (!quant_type.empty()) {
    try {
      Tensor baseline_logits = RunLogitsForTokens(baseline_model, tokens);
      Tensor quant_logits = RunLogitsForTokens(model, tokens);
      auto [max_err, mean_err] = LogitsError(baseline_logits, quant_logits);

      std::cout << "\n  logits error vs model-native:\n";
      std::cout << "    max:  " << std::scientific << std::setprecision(3)
                << max_err << "\n";
      std::cout << "    mean: " << std::scientific << std::setprecision(3)
                << mean_err << "\n";
    } catch (const std::exception& e) {
      std::cerr << "Failed to compute logits error: " << e.what() << "\n";
      return 1;
    }
  }

  return 0;
}

// ---------------------------------------------------------------------------
// Serve mode (HTTP server)
// ---------------------------------------------------------------------------
static void PrintServeUsage(const char* prog) {
  std::cout << "Usage: " << prog << " serve <model-path|dir> [options]\n"
            << "Options:\n"
            << "  --port <port>      HTTP server port (default: 8080)\n"
            << "  --tokenizer <path> Path to vocab.json tokenizer file\n"
            << "  --threads <n>      Number of threads (0 = auto)\n"
            << "  -h, --help         Show this help\n"
            << "Request body: {\"prompt\": \"...\", \"max_tokens\": 32}\n";
}

// Escape a string so it can be embedded verbatim in a JSON string literal.
// Required because generated text may contain '"', '\\' or control chars
// (e.g. newlines), which would otherwise produce invalid JSON.
static std::string JsonEscape(const std::string& text) {
  std::string out;
  out.reserve(text.size() + 16);
  for (unsigned char c : text) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          static const char kHex[] = "0123456789abcdef";
          out += "\\u00";
          out += kHex[(c >> 4) & 0x0F];
          out += kHex[c & 0x0F];
        } else {
          out += static_cast<char>(c);
        }
        break;
    }
  }
  return out;
}

// Try to read {"prompt": "..."} from a JSON request body.
// Returns true and sets `prompt` on success; returns false when the body is
// not JSON (the caller falls back to using the raw body as the prompt).
static bool ExtractPromptFromJsonBody(const std::string& body,
                                      std::string& prompt) {
  size_t key = body.find("\"prompt\"");
  if (key == std::string::npos) return false;
  size_t colon = body.find(':', key + 8);
  if (colon == std::string::npos) return false;
  size_t i = colon + 1;
  while (i < body.size() &&
         std::isspace(static_cast<unsigned char>(body[i]))) {
    ++i;
  }
  if (i >= body.size() || body[i] != '"') return false;
  ++i;  // skip opening quote

  std::string out;
  for (; i < body.size();) {
    char c = body[i];
    if (c == '"') {  // closing quote
      prompt = out;
      return true;
    }
    if (c == '\\') {
      if (i + 1 >= body.size()) return false;
      char e = body[i + 1];
      switch (e) {
        case '"': out += '"'; i += 2; break;
        case '\\': out += '\\'; i += 2; break;
        case '/': out += '/'; i += 2; break;
        case 'b': out += '\b'; i += 2; break;
        case 'f': out += '\f'; i += 2; break;
        case 'n': out += '\n'; i += 2; break;
        case 'r': out += '\r'; i += 2; break;
        case 't': out += '\t'; i += 2; break;
        case 'u': {
          // \uXXXX (BMP only; surrogate pairs not handled)
          if (i + 5 >= body.size()) return false;
          unsigned int code = 0;
          for (int j = 0; j < 4; ++j) {
            char h = body[i + 2 + j];
            code <<= 4;
            if (h >= '0' && h <= '9') {
              code |= static_cast<unsigned int>(h - '0');
            } else if (h >= 'a' && h <= 'f') {
              code |= static_cast<unsigned int>(h - 'a' + 10);
            } else if (h >= 'A' && h <= 'F') {
              code |= static_cast<unsigned int>(h - 'A' + 10);
            } else {
              return false;
            }
          }
          if (code < 0x80) {
            out += static_cast<char>(code);
          } else if (code < 0x800) {
            out += static_cast<char>(0xC0 | (code >> 6));
            out += static_cast<char>(0x80 | (code & 0x3F));
          } else {
            out += static_cast<char>(0xE0 | (code >> 12));
            out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (code & 0x3F));
          }
          i += 6;  // skip \uXXXX
          break;
        }
        default: return false;
      }
      continue;
    }
    out += c;
    ++i;
  }
  return false;  // unterminated string
}

// Try to read a non-negative integer field (e.g. "max_tokens") from a JSON
// request body. Returns false when the key is absent or not an integer.
static bool ExtractIntFieldFromJsonBody(const std::string& body,
                                        const std::string& key, int& value) {
  const std::string marker = "\"" + key + "\"";
  size_t k = body.find(marker);
  if (k == std::string::npos) return false;
  size_t colon = body.find(':', k + marker.size());
  if (colon == std::string::npos) return false;
  size_t i = colon + 1;
  while (i < body.size() &&
         std::isspace(static_cast<unsigned char>(body[i]))) {
    ++i;
  }
  size_t start = i;
  while (i < body.size() && body[i] >= '0' && body[i] <= '9') ++i;
  if (i == start) return false;
  long long v = 0;
  for (size_t j = start; j < i; ++j) {
    v = v * 10 + (body[j] - '0');
  }
  value = static_cast<int>(v);
  return true;
}

// Try to read a float field (e.g. "temperature") from a JSON request body.
// Returns false when the key is absent or not a number.
static bool ExtractFloatFieldFromJsonBody(const std::string& body,
                                          const std::string& key,
                                          float& value) {
  const std::string marker = "\"" + key + "\"";
  size_t k = body.find(marker);
  if (k == std::string::npos) return false;
  size_t colon = body.find(':', k + marker.size());
  if (colon == std::string::npos) return false;
  const char* start = body.c_str() + colon + 1;
  char* end = nullptr;
  float v = std::strtof(start, &end);
  if (end == start) return false;
  value = v;
  return true;
}

// Try to read a boolean field (e.g. "stream") from a JSON request body.
// Returns false when the key is absent or not true/false.
static bool ExtractBoolFieldFromJsonBody(const std::string& body,
                                         const std::string& key,
                                         bool& value) {
  const std::string marker = "\"" + key + "\"";
  size_t k = body.find(marker);
  if (k == std::string::npos) return false;
  size_t colon = body.find(':', k + marker.size());
  if (colon == std::string::npos) return false;
  size_t i = colon + 1;
  while (i < body.size() &&
         std::isspace(static_cast<unsigned char>(body[i]))) {
    ++i;
  }
  if (body.compare(i, 4, "true") == 0) {
    value = true;
    return true;
  }
  if (body.compare(i, 5, "false") == 0) {
    value = false;
    return true;
  }
  return false;
}

// SSE 流式生成：每生成一个 token 推一条 data: {...} 事件，
// text 字段携带到当前为止的完整生成文本（保证 UTF-8 不会被拆坏）。
static void StreamGenerate(httplib::Response& res, const LoadedModel& lm,
                           std::vector<int> tokens, int max_tokens,
                           float temperature) {
  struct StreamState {
    std::mutex m;
    std::condition_variable cv;
    std::vector<std::string> events;  // 完整的 SSE 事件
    bool done = false;
  };
  auto state = std::make_shared<StreamState>();

  // 推理在工作线程中进行，把事件推入队列；HTTP 线程负责逐个写出。
  std::thread worker([state, &lm, tokens = std::move(tokens), max_tokens,
                      temperature]() mutable {
    try {
      MiniLlamaContext ctx(&lm.model);
      SamplingParams params;
      params.temperature = temperature;
      MiniSampler sampler(params);

      MiniBatch prefill = MiniBatch::FromTokens(tokens, 0);
      Tensor logits = ForwardBatch(ctx, lm.model, prefill);

      std::vector<int> generated_ids;
      int next = 0;
      for (int i = 0; i < max_tokens; ++i) {
        if (i > 0) {
          MiniBatch decode_batch = MiniBatch::Single(
              tokens.back(), static_cast<int>(tokens.size()) - 1);
          logits = ForwardBatch(ctx, lm.model, decode_batch);
        }
        next = sampler.Sample(logits, params);
        if (next == lm.tokenizer->eos_id()) break;
        generated_ids.push_back(next);
        tokens.push_back(next);

        std::string text = lm.tokenizer->Decode(generated_ids);
        std::string event =
            "data: {\"text\":\"" + JsonEscape(text) + "\"}\n\n";
        {
          std::lock_guard<std::mutex> lock(state->m);
          state->events.push_back(std::move(event));
        }
        state->cv.notify_one();
      }

      std::string done_event =
          "data: {\"text\":\"" +
          JsonEscape(lm.tokenizer->Decode(generated_ids)) +
          "\",\"done\":true}\n\n";
      {
        std::lock_guard<std::mutex> lock(state->m);
        state->events.push_back(std::move(done_event));
      }
    } catch (const std::exception& e) {
      std::string err_event = "data: {\"error\":\"" + JsonEscape(e.what()) +
                              "\",\"done\":true}\n\n";
      {
        std::lock_guard<std::mutex> lock(state->m);
        state->events.push_back(std::move(err_event));
      }
    }
    {
      std::lock_guard<std::mutex> lock(state->m);
      state->done = true;
    }
    state->cv.notify_all();
  });
  worker.detach();

  res.set_header("Cache-Control", "no-cache");
  res.set_header("Connection", "keep-alive");
  res.set_chunked_content_provider(
      "text/event-stream",
      [state](size_t /*offset*/, httplib::DataSink& sink) {
        std::unique_lock<std::mutex> lk(state->m);
        while (state->events.empty() && !state->done) {
          state->cv.wait(lk);
        }
        if (!state->events.empty()) {
          std::string event = std::move(state->events.front());
          state->events.erase(state->events.begin());
          lk.unlock();
          if (!sink.write(event.data(), event.size())) {
            return false;  // 客户端已断开
          }
          return true;
        }
        lk.unlock();
        sink.done();
        return false;
      });
}

// 处理 POST /v1/generate：解析请求 -> 推理 -> 返回 JSON（或 SSE 流）。
// 支持请求体：{"prompt": "...", "max_tokens": N, "temperature": 0~2,
//             "stream": true|false}
// 所有异常都转换为 500 + {"error": ...}，不会挂起连接。
static void HandleGenerate(const httplib::Request& req, httplib::Response& res,
                           const LoadedModel& lm) {
  // 1. 解析 prompt：优先查询参数 ?prompt=，其次 JSON body
  //    {"prompt": "..."}，最后兼容纯文本 body。
  std::string prompt = req.get_param_value("prompt");
  if (prompt.empty()) {
    std::string json_prompt;
    if (ExtractPromptFromJsonBody(req.body, json_prompt)) {
      prompt = json_prompt;
    } else {
      prompt = req.body;
    }
  }
  if (prompt.empty()) {
    res.status = 400;
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_content("{\"error\":\"missing prompt\"}", "application/json");
    return;
  }

  try {
    std::vector<int> tokens = lm.tokenizer->Encode(prompt);
    if (tokens.empty()) {
      res.status = 400;
      res.set_header("Access-Control-Allow-Origin", "*");
      res.set_content("{\"error\":\"tokenization failed\"}", "application/json");
      return;
    }

    // 2. 生成数量：默认 32，支持 body 里 {"max_tokens": N}，
    //    并受 max_seq_len 限制，避免位置越界报错。
    int max_tokens = 32;
    ExtractIntFieldFromJsonBody(req.body, "max_tokens", max_tokens);
    int max_allowed =
        lm.model.config.max_seq_len - static_cast<int>(tokens.size());
    if (max_allowed < 1) {
      res.status = 400;
      res.set_header("Access-Control-Allow-Origin", "*");
      res.set_content("{\"error\":\"prompt longer than max_seq_len\"}",
                      "application/json");
      return;
    }
    if (max_tokens > max_allowed) max_tokens = max_allowed;
    if (max_tokens < 1) max_tokens = 1;

    // 3. temperature：0 = 贪心，默认 0（与旧行为一致）
    float temperature = 0.0f;
    ExtractFloatFieldFromJsonBody(req.body, "temperature", temperature);
    if (!std::isfinite(temperature) || temperature < 0.0f ||
        temperature > 2.0f) {
      temperature = 0.0f;
    }

    // 4. stream：true 时走 SSE 逐 token 输出
    bool stream = false;
    ExtractBoolFieldFromJsonBody(req.body, "stream", stream);

    res.set_header("Access-Control-Allow-Origin", "*");
    if (stream) {
      StreamGenerate(res, lm, std::move(tokens), max_tokens, temperature);
      return;
    }

    // 非流式：一次性返回完整 JSON
    MiniLlamaContext ctx(&lm.model);
    SamplingParams params;
    params.temperature = temperature;
    MiniSampler sampler(params);

    // Prefill：一次处理整个 prompt
    MiniBatch prefill = MiniBatch::FromTokens(tokens, 0);
    Tensor logits = ForwardBatch(ctx, lm.model, prefill);

    std::vector<int> generated_ids;
    int next = 0;
    for (int i = 0; i < max_tokens; ++i) {
      if (i > 0) {
        // Decode：单 token 前向，位置为当前序列末尾
        MiniBatch decode_batch =
            MiniBatch::Single(tokens.back(), static_cast<int>(tokens.size()) - 1);
        logits = ForwardBatch(ctx, lm.model, decode_batch);
      }
      next = sampler.Sample(logits, params);
      if (next == lm.tokenizer->eos_id()) break;
      generated_ids.push_back(next);
      tokens.push_back(next);
    }

    std::string output_text = lm.tokenizer->Decode(generated_ids);

    // 5. 返回 JSON（生成文本需转义，否则含引号/换行时 JSON 非法）
    std::string json = "{\"generated_text\":\"" + JsonEscape(output_text) + "\"}";
    res.set_content(json, "application/json");
  } catch (const std::exception& e) {
    res.status = 500;
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_content("{\"error\":\"" + JsonEscape(e.what()) + "\"}",
                    "application/json");
  }
}

static int RunServe(int argc, char** argv) {
  if (argc >= 3 && (std::strcmp(argv[2], "-h") == 0 ||
                    std::strcmp(argv[2], "--help") == 0)) {
    PrintServeUsage(argv[0]);
    return 0;
  }
  if (argc < 3) {
    std::cerr << "Missing model directory.\n";
    PrintServeUsage(argv[0]);
    return 1;
  }

  std::string model_dir = argv[2];
  int port = 8080;
  std::string tokenizer_path;
  int n_threads = 0;
  BackendConfig backend_config;

  // 解析参数
  for (int i = 3; i < argc; ++i) {
    if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
      if (!ParseIntArg(argv[++i], port) || port <= 0 || port > 65535) {
        std::cerr << "Invalid --port value.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
      tokenizer_path = argv[++i];
    } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      if (!ParseIntArg(argv[++i], n_threads) || n_threads < 0) {
        std::cerr << "Invalid --threads value.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
      if (!ParseBackendArg(argv[++i], backend_config)) return 1;
    } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
      if (!ParseDeviceArg(argv[++i], backend_config)) return 1;
    } else if (std::strcmp(argv[i], "-h") == 0 ||
               std::strcmp(argv[i], "--help") == 0) {
      PrintServeUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << argv[i] << "\n";
      return 1;
    }
  }

  if (!ValidateBackendOrPrint(backend_config)) return 1;

  // 加载模型（只加载一次，全局持有）
  static LoadedModel lm = LoadModelAndTokenizer(model_dir, "", tokenizer_path);
  if (!lm.model.loaded) {
    std::cerr << "Failed to load model: " << lm.model.load_error << "\n";
    return 1;
  }
  if (!lm.tokenizer) {
    std::cerr << "Failed to load tokenizer.\n";
    return 1;
  }
  SetThreadCount(n_threads);

  // 启动 HTTP 服务器
  httplib::Server svr;
  // 放宽超时：SSE 流式生成时，写超时过短会导致连接被中途断开
  svr.set_read_timeout(120, 0);
  svr.set_write_timeout(300, 0);

  // 健康检查：返回服务状态 + 模型信息（网页界面用）。
  // lm 是 RunServe 内的 static 变量，lambda 内可直接使用，无需捕获。
  svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
    std::ostringstream oss;
    oss << "{\"status\":\"ok\",\"model\":{\"n_layers\":"
        << lm.model.config.n_layers << ",\"dim\":" << lm.model.config.dim
        << ",\"vocab_size\":" << lm.model.config.vocab_size
        << ",\"max_seq_len\":" << lm.model.config.max_seq_len << "}}";
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_content(oss.str(), "application/json");
  });

  // 网页界面：把 web/ 目录挂到根路径（兼容从 build/ 或项目根目录启动）
  std::string web_dir = "./web";
  if (!svr.set_mount_point("/", web_dir)) {
    web_dir = "../web";
    if (!svr.set_mount_point("/", web_dir)) {
      std::cerr << "Warning: web UI directory not found (tried ./web and "
                   "../web). Only the API is available.\n";
    }
  }

  // 定义 POST 接口：/v1/generate，body 形如
  // {"prompt":"...","max_tokens":N,"temperature":0.8,"stream":true}
  svr.Post("/v1/generate", [&](const httplib::Request& req,
                               httplib::Response& res) {
    auto t0 = std::chrono::steady_clock::now();
    HandleGenerate(req, res, lm);
    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0)
            .count();
    std::cout << "[serve] " << req.method << " " << req.path << " -> "
              << res.status << " (" << elapsed_ms << " ms)\n";
  });

  std::cout << "Model loaded: " << lm.model.config.n_layers
            << " layers, dim " << lm.model.config.dim << ", vocab "
            << lm.model.config.vocab_size << ", max_seq_len "
            << lm.model.config.max_seq_len << std::endl;
  std::cout << "Web UI:    http://localhost:" << port << "/" << std::endl;
  std::cout << "API:       http://localhost:" << port
            << "/v1/generate (POST)" << std::endl;
  std::cout << "Server listening on 0.0.0.0:" << port << std::endl;
  if (!svr.listen("0.0.0.0", port)) {
    std::cerr << "Failed to listen on 0.0.0.0:" << port
              << " (port already in use?).\n";
    return 1;
  }
  return 0;
}

}  // namespace mini_llama

int main(int argc, char** argv) {
  if (argc < 2) {
    mini_llama::PrintGlobalUsage(argv[0]);
    return 1;
  }

  std::string command = argv[1];
  if (command == "generate") {
    return mini_llama::RunGenerate(argc, argv);
  } else if (command == "run") {
    return mini_llama::RunChat(argc, argv);
  } else if (command == "inspect") {
    return mini_llama::RunInspect(argc, argv);
  } else if (command == "bench") {
    return mini_llama::RunBench(argc, argv);
  } else if (command == "inspect-gguf") {
    return mini_llama::RunInspectGguf(argc, argv);
  } else if (command == "-h" || command == "--help") {
    mini_llama::PrintGlobalUsage(argv[0]);
    return 0;
  } else if (command == "serve") {   // <--- 新增这一行
    return mini_llama::RunServe(argc, argv);
  } else if (!command.empty() && command[0] == '-') {
    // Backward compatibility: default to generate mode when first arg is a
    // flag. Shift argv[1..] down by inserting "generate" at position 1.
    std::vector<char*> shifted_argv;
    shifted_argv.reserve(argc + 1);
    shifted_argv.push_back(argv[0]);
    shifted_argv.push_back(const_cast<char*>("generate"));
    for (int i = 1; i < argc; ++i) {
      shifted_argv.push_back(argv[i]);
    }
    return mini_llama::RunGenerate(argc + 1, shifted_argv.data());
  } else {
    std::cerr << "Unknown command: " << command << "\n";
    mini_llama::PrintGlobalUsage(argv[0]);
    return 1;
  }
}
