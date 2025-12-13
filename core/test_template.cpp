
#include "include/gpu/backend.hpp"
#include <iostream>

using namespace CaptionEngine::GPU;

GPUResult<int> test_func() { return GPUError{"Test error"}; }

int main() {
  auto res = test_func();
  if (!res) {
    std::cout << "Error: " << res.error().message << std::endl;
  }
  return 0;
}
