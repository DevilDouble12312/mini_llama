// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/backend.h"

#include <stdexcept>
#include <string>

namespace mini_llama {

bool ParseBackendKind(const std::string& text, BackendKind& kind) {
  if (text == "cpu") {
    kind = BackendKind::kCpu;
    return true;
  }
  return false;
}

std::string BackendKindName(BackendKind kind) {
  switch (kind) {
    case BackendKind::kCpu:
      return "cpu";
  }
  return "unknown";
}

void ValidateBackend(const BackendConfig& config) {
  switch (config.kind) {
    case BackendKind::kCpu:
      return;
  }
  throw std::runtime_error("unsupported backend");
}

std::string BackendExecutionNote(const BackendConfig& config) {
  switch (config.kind) {
    case BackendKind::kCpu:
      return "execution: cpu";
  }
  return "execution: unknown";
}

}  // namespace mini_llama
