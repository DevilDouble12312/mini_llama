cmake_minimum_required(VERSION 3.17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=x86-64-v3" CACHE STRING "" FORCE)

project(mini-llama CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include_directories(${CMAKE_SOURCE_DIR}/include)

include_directories(${CMAKE_SOURCE_DIR})

include(FetchContent)
FetchContent_Declare(
    googletest
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/../third_party/googletest
)
FetchContent_MakeAvailable(googletest)

set(TENSOR_SOURCES
  src/tensor.cpp
  src/matmul_optimized.cpp
  src/ops.cpp
  src/thread_pool.cpp
  src/tokenizer.cpp
  src/json_vocab_tokenizer.cpp
  src/bpe_tokenizer.cpp
  src/kv_cache.cpp
  src/context.cpp
  src/radix_tree.cpp
  src/request_context.cpp
  src/loader.cpp
  src/forward.cpp
  src/quantized_tensor.cpp
  src/quant.cpp
  src/sampler.cpp
  src/batch.cpp
  src/chat.cpp
  src/gguf.cpp
  src/gguf_loader.cpp
  src/gguf_tokenizer.cpp
  src/terminal.cpp
  src/prompt_builder.cpp
  src/backend.cpp
  src/debug.cpp)

file(GLOB MY_TEST_SOURCES CONFIGURE_DEPENDS tests/*.cc)

set(MY_TEST_SOURCES
  ${TENSOR_SOURCES}
  ${MY_TEST_SOURCES}
)


add_executable(tensor_my_tests ${MY_TEST_SOURCES})

target_link_libraries(tensor_my_tests PRIVATE GTest::gtest_main)

add_executable(mini-llama ${TENSOR_SOURCES} src/main.cpp)

target_link_libraries(mini-llama PRIVATE)

set(CMAKE_CXX_FLAGS_RELEASE "-O2")

file(COPY ${CMAKE_SOURCE_DIR}/models DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/web DESTINATION ${CMAKE_BINARY_DIR})