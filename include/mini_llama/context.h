#pragma once 

#include "mini_llama/kv_cache.h"
#include "mini_llama/model.h"
#include <vector>


namespace mini_llama {

    struct MiniLlamaContext{

        const MiniLlamaModel* model = nullptr;
        KvCache kv_cache;
        int pos=0;

        std::vector<int> token_history;

        int n_prefill_tokens = 0;
        int n_decode_tokens = 0;

        MiniLlamaContext() = default;
        MiniLlamaContext(const MiniLlamaContext&) = delete;
        MiniLlamaContext& operator=(const MiniLlamaContext&) = delete; 
        MiniLlamaContext(MiniLlamaContext&&) noexcept = default;
        MiniLlamaContext& operator=(MiniLlamaContext&&) noexcept = default;
        explicit MiniLlamaContext(const MiniLlamaModel* model);
    };
}