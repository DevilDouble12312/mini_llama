#include "mini_llama/forward.h"
#include "mini_llama/context.h"
#include "mini_llama/kv_cache.h"
#include "mini_llama/model.h"
#include "mini_llama/ops.h"
#include "mini_llama/tensor.h"


#include "mini_llama/thread_pool.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mini_llama {

    //isvalid
    static void ValidateForwardInputs(const MiniLlamaContext& ctx, const MiniLlamaModel& model, int token) {
        const ModelConfig& c = model.config;
        if(!model.loaded) {
            throw std::runtime_error("ForwardToken called with an unloaded model");
        }

        if(token < 0 || token >= c.vocab_size) {
            throw std::out_of_range("ForwardToken token id out of range");
        }

        if(ctx.pos < 0 || ctx.pos >= c.max_seq_len) {
            throw std::out_of_range("ForwardToken position out of range");
        }

        if(model.layers.size() != static_cast<size_t>(c.n_layers)) {
            throw std::runtime_error("ForwardToken layer count does not match model config");
        }

        if(model.token_embedding.size() < static_cast<size_t>(c.vocab_size * c.dim)) {
            throw std::runtime_error("ForwardToken token embedding tensor is smaller than config");
        }
    }

    static void RequireShape(const Tensor& t, const std::vector<int>& expected, const char* caller) {
        t.AssertShape(expected, caller);
    }

    //EmbedToken
    static Tensor EmbedToken(const MiniLlamaModel& model, int token_id) {
        int dim =  model.config.dim;
        Tensor x({dim}, 0.0f);
        for(int i = 0; i < dim; ++i) {
            x.data[i] = model.token_embedding.data[token_id * dim + i];
        }
        return x;
    }

    //ForwardRmsNorm
    static Tensor ForwardRmsNorm(const MiniLlamaModel& model, const Tensor& x, const Tensor& weight, float eps) {
        return RmsNorm(x, weight, eps);
    }

    //ForwardLinear
    static Tensor ForwardLinear(const MiniLlamaModel& model, const std::string& name, const Tensor& x, const QuantizedTensor& weight) {
        return Linear(x, weight);
    }

    //AddOptionalBias
    static Tensor AddOptionalBias(const Tensor& x, const Tensor& bias, const char* caller) {
        if(bias.data.empty()) {
            return x;
        }
        // just one token, so dim is one
        if(x.num_dims() != 1 || bias.num_dims() != 1 ||x.shape[0] != bias.shape[0]) {
            throw std::runtime_error(std::string(caller) +
                             ": bias shape mismatch x=" + x.ShapeStringShort() +
                             " bias=" + bias.ShapeStringShort());
        
        }
        Tensor y = x;
        for (int i = 0; i < x.shape[0]; ++i) {
            y.data[i] += bias.data[i];
        }
        return y;
    }
    //ForwardQkvProjection
    void ForwardQkvProjection(const MiniLlamaModel& model, const std::string& layer_prefix, const Tensor& h, const LayerWeights& lw, Tensor& q_flat, Tensor& k_flat, Tensor& v_flat) {
        const std::string q_name = layer_prefix + "wq";
        const std::string k_name = layer_prefix + "wk";
        const std::string v_name = layer_prefix + "wv";

        q_flat = AddOptionalBias(ForwardLinear(model, q_name, h, lw.wq), lw.bq, "forward_layer q");
        k_flat = AddOptionalBias(ForwardLinear(model, k_name, h, lw.wk), lw.bk, "forward_layer k");
        v_flat = AddOptionalBias(ForwardLinear(model, v_name, h, lw.wv), lw.bv, "forward_layer v");
    }

    //ForwardRope
    void ForwardRope(const MiniLlamaModel&model, Tensor& q, Tensor& k, int pos, float theta, RopeType rope_type) {
        Rope(q, k, pos, theta, rope_type);
    }

    //GQA head mapping
    static int MapQHeadToKvHead(int q_head, int n_heads, int n_kv_heads) {
        return q_head / (n_heads / n_kv_heads);
    }

    //ForwardSoftmax
    static Tensor ForwardSoftmax(const MiniLlamaModel& model, const Tensor& x) {
        return Softmax(x);
    }


    Tensor AttentionForward(const MiniLlamaModel& model, const Tensor& q, const Tensor& k, const Tensor& v,
                            int pos, int layer, KvCache& kv_cache,
                            int n_heads, int n_kv_heads, int head_dim) {
        //write the token (done) k v
        kv_cache.Write(layer, pos, k, v);

        Tensor attn_out({n_heads, head_dim}, 0.0f);


        //scale缩放因子
        float scale = 1.0f/ std::sqrt(static_cast<float>(head_dim));

        ParallelFor(n_heads, [&](int begin, int end) {
            for(int h = begin; h < end; ++h) {
                //1,解析 GQA映射, 获取对应的kv 头索引
                int kv_head = MapQHeadToKvHead(h, n_heads, n_kv_heads);//分成8组,每组4人,h区分Q头

                //2,内积计算,多头Q与所有历史位置K 计算相似度得分
                std::vector<float> scores_data(pos + 1, 0.0f);
                for(int t = 0; t <= pos; ++t) { //遍历所有token,计算分数
                    const float* k_ptr = kv_cache.KeyPtr(layer, t, kv_head);
                    float dot = 0.0f;//内积

                    for(int d = 0; d < head_dim; ++d) { //遍历向量本身
                        dot += q.data[h* head_dim + d] * k_ptr[d]; //将Q与当前遍历的token的k做点积
                    }
                    scores_data[t] = dot * scale;
                }
                //Softmax只认Tensor, 将数据保存在Tensor中,用move高效转移
                Tensor scores({pos+1}, 0.0f);
                scores.data = std::move(scores_data);
                Tensor probs = ForwardSoftmax(model, scores);

                //3,特征加权, 使用概率分布对历史所有的 v 进行加权求和
                for(int d = 0; d < head_dim; ++d) {
                    float val = 0.0f;
                    for(int t = 0; t <= pos; ++t) {
                        const float_t* v_ptr = kv_cache.ValuePtr(layer, t, kv_head);
                        val += probs.data[t] * v_ptr[d];
                    }
                    attn_out.data[h * head_dim + d] = val;
                }
            }
        });
            return attn_out;
        }

    Tensor ForwardAdd(const MiniLlamaModel& model, const Tensor& a, const Tensor& b) {
        if(a.shape != b.shape) {
            throw std::runtime_error("forward_add: shape mismatch " +
                           a.ShapeStringShort() + " vs " +
                           b.ShapeStringShort());
        }
        Tensor y (a.shape, 0.0f);
        for(size_t i = 0; i < a.size(); ++i) {
            y.data[i] = a.data[i] + b.data[i];
        }
        return y;
    }

    static Tensor ForwardSwiGlu(const MiniLlamaModel& model, const Tensor& gate,
                            const Tensor& up) {
        return SwiGlu(gate, up);
    }

    static Tensor FfnForward(const MiniLlamaModel& model, const std::string& layer_prefix, const Tensor& h,
                         const LayerWeights& lw) {
        Tensor gate = ForwardLinear(model, layer_prefix + "w_gate", h, lw.w_gate);
        Tensor up = ForwardLinear(model, layer_prefix + "w_up", h, lw.w_up);
        Tensor ff =ForwardLinear(model, layer_prefix + "w_down",ForwardSwiGlu(model, gate, up), lw.w_down);
        return ff;
    }
    //ForwardLayer
    Tensor ForwardLayer(MiniLlamaContext& ctx, const MiniLlamaModel& model, const Tensor& x, int layer, const LayerWeights& lw, const ModelConfig& c) {
        int dim = c.dim;
        int n_heads = c.n_heads;
        int n_kv_heads = c.n_kv_heads;
        int head_dim = c.head_dim;
        int pos = ctx.pos;
        const std::string layer_prefix = "layers." + std::to_string(layer) + "."; 

        //Pre-Norm
        Tensor h = ForwardRmsNorm(model, x, lw.attention_norm, c.rms_norm_eps);

        //Q, K, V
        Tensor q_flat;
        Tensor k_flat;
        Tensor v_flat;
        ForwardQkvProjection(model, layer_prefix, h, lw, q_flat, k_flat, v_flat);

        Tensor q = q_flat.ReshapeChecked({n_heads, head_dim}, "forward_layer q");
        Tensor k = k_flat.ReshapeChecked({n_kv_heads, head_dim}, "forward_layer k");
        Tensor v = v_flat.ReshapeChecked({n_kv_heads, head_dim}, "forward_layer v");

        ForwardRope(model, q, k, pos, c.rope_theta, c.rope_type);

        //Attention: compute + read/write KV cache
        Tensor attn_out = AttentionForward(model, q, k, v, pos, layer, ctx.kv_cache,
                                   n_heads, n_kv_heads, head_dim);

        //Project and residual
        //多头融合,残差连接 1
        Tensor attn_out_flat = attn_out.ReshapeChecked({1, n_heads * head_dim}, "forward_layer atten_out");
        Tensor attn_proj = ForwardLinear(model, layer_prefix + "wo", attn_out_flat, lw.wo);
        Tensor attn_proj_1d =attn_proj.ReshapeChecked({dim}, "forward_layer attn_proj 1d");

        //residual 1
        Tensor x_attn = ForwardAdd(model, x, attn_proj_1d);

        //-----------one half end---------we get x_attn    

        // ---- FFN sublayer ----
        Tensor h2 = ForwardRmsNorm(model, x_attn, lw.ffn_norm, c.rms_norm_eps);
        Tensor ff = FfnForward(model, layer_prefix, h2, lw);
        RequireShape(ff, {dim}, "forward_layer ffn");
        return ForwardAdd(model, x_attn, ff);    
    }

    //ComputeLogits
    Tensor ComputeLogits(Tensor x, const MiniLlamaModel& model, const ModelConfig& c) {
        Tensor normed = ForwardRmsNorm(model, x, model.final_norm, c.rms_norm_eps);
        Tensor logits_flat = ForwardLinear(model, "lm_head", normed, model.lm_head);
        return logits_flat.ReshapeChecked({c.vocab_size}, "compute_logits");
    }
    //---------ForwardToken------------
    Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model, int token) {
        //Isvalid
        ValidateForwardInputs(ctx, model, token);

        const ModelConfig& c = model.config;
        int n_layers = c.n_layers;

         // 1. Embedding lookup
        Tensor x = EmbedToken(model, token);

         // 2. Transformer layers
        for(int layer = 0; layer < n_layers; ++layer) {
            x = ForwardLayer(ctx, model, x, layer, model.layers[layer], c);
        }

        // 3. Final norm + logits
        Tensor logits = ComputeLogits(x, model, c);

        return logits;

    }
}