// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/debug.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <utility>
#include <vector>

#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/sampler.h"

namespace mini_llama {

// ---------------------------------------------------------------------------
// Debug dump helpers
// ---------------------------------------------------------------------------
void DumpTensorShape(const Tensor& t, const std::string& name) {
  std::cout << "  " << name << " shape: " << t.ShapeStringShort() << "\n";
}

void DumpLogitsTopK(const Tensor& logits, int k) {
  if (k <= 0) {
    return;
  }
  std::vector<std::pair<float, int>> indexed;
  indexed.reserve(logits.size());
  for (size_t i = 0; i < logits.size(); ++i) {
    indexed.emplace_back(logits.data[i], static_cast<int>(i));
  }
  std::partial_sort(indexed.begin(),
                    indexed.begin() + std::min(k, static_cast<int>(indexed.size())),
                    indexed.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
  int n = std::min(k, static_cast<int>(indexed.size()));
  std::cout << "  top-" << n << " logits:";
  for (int i = 0; i < n; ++i) {
    std::cout << " [" << indexed[i].second << "]=" << indexed[i].first;
  }
  std::cout << "\n";
}

void DumpKvCacheInfo(const KvCache& cache, int current_pos) {
  std::cout << "  kv cache keys shape: " << cache.keys.ShapeStringShort() << "\n";
  std::cout << "  kv cache values shape: " << cache.values.ShapeStringShort()
            << "\n";
  if (current_pos >= 0) {
    std::cout << "  kv cache current position: " << current_pos << "\n";
  }
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------
BenchmarkResult RunBenchmark(const MiniLlamaModel& model,
                             const std::vector<int>& prompt_tokens,
                             int n_predict, unsigned int seed, bool verbose) {
  BenchmarkResult result;

  MiniLlamaContext ctx(&model);
  SamplingParams sampling_params;
  sampling_params.temperature = 0.0f;  // greedy
  sampling_params.top_k = 0;
  sampling_params.seed = seed;
  MiniSampler sampler(sampling_params);

  if (verbose) {
    DumpTensorShape(model.token_embedding, "token_embedding");
    DumpTensorShape(model.final_norm, "final_norm");
  }

  // Prefill
  Tensor logits;
  {
    MiniBatch prefill = MiniBatch::FromTokens(prompt_tokens, 0);
    auto start = std::chrono::steady_clock::now();
    logits = ForwardBatch(ctx, model, prefill);
    auto end = std::chrono::steady_clock::now();
    result.n_prompt_tokens = static_cast<int>(prompt_tokens.size());
    result.prefill_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
  }
  if (verbose) {
    DumpLogitsTopK(logits, 5);
  }

  // Decode loop
  std::vector<int> generated;
  generated.reserve(static_cast<size_t>(n_predict));
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < n_predict; ++i) {
    int next_token = sampler.Sample(logits, sampling_params);
    generated.push_back(next_token);
    int pos = static_cast<int>(prompt_tokens.size() + generated.size() - 1);
    MiniBatch decode = MiniBatch::Single(next_token, pos);
    logits = ForwardBatch(ctx, model, decode);
    if (verbose) {
      DumpLogitsTopK(logits, 5);
      DumpKvCacheInfo(ctx.kv_cache, pos);
    }
  }
  auto end = std::chrono::steady_clock::now();
  result.n_decode_tokens = static_cast<int>(generated.size());
  result.n_generated_tokens = static_cast<int>(generated.size());
  result.decode_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  return result;
}

}  // namespace mini_llama
