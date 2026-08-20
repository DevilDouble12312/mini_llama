#pragma once

#include <cstdio>
#include <vector>
#include <map>
#include <memory>

namespace mini_llama{

    class RadixTree{
        public:
            RadixTree();
            ~RadixTree();

            RadixTree(const RadixTree&) = delete;
            RadixTree& operator=(const RadixTree&) = delete;
            RadixTree(RadixTree&&) noexcept;
            RadixTree& operator=(RadixTree&&) noexcept;

            void Clear();
            bool empty() const;
            size_t size() const;

            void Insert(const std::vector<int>& tokens);
            size_t LongestPrefix(const std::vector<int>& tokens) const;
        private:
            struct Node{
                std::vector<int> key;
                bool terminal = false;
                std::map<int, std::unique_ptr<Node>> children;
            };

             void InsertInto(Node* parent, std::vector<int> suffix);

            static size_t CommonPrefixLength(const std::vector<int>& a, size_t a_offset,
                                   const std::vector<int>& b);


            Node root_;
            size_t terminal_count_ = 0;
    };
} //mini_llama