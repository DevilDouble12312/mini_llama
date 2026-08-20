#pragma once

#include <map>
#include <string>
#include <vector>

#include "mini_llama/model.h"

namespace mini_llama {

    struct TensorInfo {
        std::string name;
        std::vector<int> shape;
        std::string dtype;
        size_t offset = 0;
        size_t byte_size = 0;
    };

    struct TokenizerInfo {
        std::string type = "ascii";
        std::string path;
        int bos_id = 1;
        int eos_id = 2;
        int unk_id = 0;
    };



    struct ModelManifest {
        ModelConfig config;
        TokenizerInfo tokenizer;
        std::vector<TensorInfo> tensors;
    };

    ModelManifest ParseManifest(const std::string& config_path);

    MiniLlamaModel LoadModel(const std::string& config_path, const std::string& weights_path);

    bool InspectModel(const std::string& config_path);


}