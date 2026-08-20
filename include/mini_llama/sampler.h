#pragma once

#include "mini_llama/tensor.h"
#include <random>

namespace mini_llama {

    struct SamplingParams{
        float temperature = 0.0f;
        int top_k = 0;
        unsigned int seed = 0;
    };

    class MiniSampler {
    public:
        explicit MiniSampler(unsigned int seed = 0);
        explicit MiniSampler(const SamplingParams& params);

        static int SampleGreedy(const Tensor& logits);

        int Sample(const Tensor& logits, const SamplingParams& params);

        int SampleTemperature(const Tensor& logits, float temperature);

        int SampleTopK(const Tensor& logits, float temperature, int top_k);
    private:
        std::mt19937 rng_;

    };


    int SampleGreedy(const Tensor& logits);

} //namespace mini_llama