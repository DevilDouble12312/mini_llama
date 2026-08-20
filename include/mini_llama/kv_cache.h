#pragma once

#include "mini_llama/tensor.h"
#include <cstring>


namespace mini_llama {
    struct KvCache {
        Tensor keys;
        Tensor values;

        KvCache() = default;
        KvCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim);

        void Write(int layer, int pos, const Tensor& k, const Tensor& v);

        const float* KeyPtr(int layer, int pos, int kv_head) const;
        const float* ValuePtr(int layer, int pos, int kv_head) const;

    };

} //namespace mini_llama