#pragma once

#include <thread>
#include <functional>

namespace mini_llama{
    int GetThreadCount();

    void SetThreadCount(int count);

    void ParallelFor(int n, const std::function<void(int begin, int end)>& fn);

}// mini_llama