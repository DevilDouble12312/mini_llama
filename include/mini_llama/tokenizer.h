#pragma once 

#include <vector>
#include <string>
#include <memory>
#include <stdint.h>
#include <unordered_map>
#include <algorithm>
#include <map>

namespace mini_llama {
    class ITokenizer {
        public: 
            virtual ~ITokenizer() = default;

            virtual std::vector<int> Encode(const std::string& text) const = 0;

            virtual std::string DecodeToken(int token) const = 0;

            virtual std::string Decode(const std::vector<int>& tokens) const = 0;

            virtual int vocab_size() const = 0;
            virtual int bos_id() const = 0;
            virtual int eos_id() const = 0;
            virtual int unk_id() const = 0;
    };

    // -----------ASCII tokenizer----------- 
    class AsciiTokenizer : public ITokenizer {
        public:
            static constexpr int kVocabSize = 128;
            
            AsciiTokenizer() = default;

            std::vector<int> Encode(const std::string& text) const override;

            std::string DecodeToken(int token) const override;

            std::string Decode(const std::vector<int>& tokens) const override;

            int vocab_size() const override { return kVocabSize; }
            int bos_id() const override { return 1; }
            int eos_id() const override { return 2; }
            int unk_id() const override { return 0; }
    };

    //-----------JSON------------

    class JsonVocabTokenizer : public ITokenizer {
        public:
            explicit JsonVocabTokenizer(const std::string& vocab_path);

            std::vector<int> Encode(const std::string& text) const override;
            std::string DecodeToken(int token) const override;
            std::string Decode(const std::vector<int>& tokens) const override;

            int vocab_size() const override { return vocab_size_; }
            int bos_id() const override { return bos_id_; }
            int eos_id() const override { return eos_id_; }
            int unk_id() const override { return unk_id_; }

        private:
            struct VocabEntry {
                int id = 0;
                std::string content;
                bool special = false;
            };

            int vocab_size_ = 0;
            int bos_id_ = 1;
            int eos_id_ = 2;
            int unk_id_ = 0;
            std::vector<VocabEntry> id_to_entry_;
            std::vector<std::pair<std::string, int>> content_to_id_;
    };

    //------------BPE------------
    class BpeTokenizer : public ITokenizer {
        public:
            BpeTokenizer() = default;
            ~BpeTokenizer() = default;

            bool Load(const std::string& vocab_path, const std::string& merges_path, const std::string& special_path);

            std::vector<int> Encode(const std::string& text) const override;
            std::string DecodeToken(int token) const override;
            std::string Decode(const std::vector<int>& tokens) const override;

            int vocab_size() const override {
                return static_cast<int>(id_to_token_.size());
            }
            int bos_id() const override { return bos_id_; }
            int eos_id() const override { return eos_id_; }
            int unk_id() const override { return unk_id_; }
        private:
            std::unordered_map<std::string, int> vocab_;
            std::vector<std::string> id_to_token_;
            std::map<std::pair<std::string, std::string>, int> merge_ranks_;
            int bos_id_ = -1;
            int eos_id_ = -1;
            int unk_id_ = -1;

            std::vector<std::pair<std::string, int>> special_tokens_;

            std::vector<std::string> b2u_;                  // byte -> unicode string
            std::unordered_map<std::string, uint8_t> u2b_;  // unicode string -> byte

            void BuildByteMappings();

    };
    //factory
    std::unique_ptr<ITokenizer> CreateTokenizer(const std::string& vocab_path);

    std::unique_ptr<ITokenizer> CreateBpeTokenizer(const std::string& vocab_path, 
                        const std::string& merges_path, const std::string& special_path);
} // namespace mini_llam
