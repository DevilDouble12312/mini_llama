#pragma once

#include "mini_llama/context.h"
#include "mini_llama/model.h"
#include "mini_llama/tensor.h"
#include "mini_llama/batch.h"

namespace mini_llama {
    Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model, int token);

    Tensor ForwardBatch(MiniLlamaContext& ctx, const MiniLlamaModel& model, const MiniBatch& batch);


    
} // mini_llama