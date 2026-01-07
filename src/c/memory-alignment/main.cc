#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>

constexpr size_t ARRAY_SIZE = 1024 * 1024;
constexpr size_t ITERATIONS = 100;

alignas(64) static char buffer[ARRAY_SIZE * sizeof(int) + 64];

__attribute__((noinline)) int64_t sum_array(const int* arr, size_t n) {
  int64_t sum = 0;
  for (size_t i = 0; i < n; ++i) {
    sum += arr[i];
  }
  return sum;
}

double benchmark(size_t offset) {
  int* arr = reinterpret_cast<int*>(buffer + offset);
  for (size_t i = 0; i < ARRAY_SIZE; ++i) {
    arr[i] = static_cast<int>(i & 0xFF);
  }

  auto start = std::chrono::high_resolution_clock::now();
  volatile int64_t result = 0;
  for (size_t i = 0; i < ITERATIONS; ++i) {
    result = sum_array(arr, ARRAY_SIZE);
  }
  auto end = std::chrono::high_resolution_clock::now();
  (void)result;

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  return duration.count() / 1000.0;
}

int main() {
  std::cout << "Memory Alignment Performance Benchmark\n";
  std::cout << "Array size: " << ARRAY_SIZE << " integers\n";
  std::cout << "Iterations: " << ITERATIONS << "\n\n";

  std::cout << std::setw(12) << "Offset" << std::setw(15) << "Time (ms)" << std::setw(15) << "Aligned to" << "\n";
  std::cout << std::string(42, '-') << "\n";

  size_t offsets[] = {0, 1, 2, 4, 8, 16, 32};
  for (size_t offset : offsets) {
    double time = benchmark(offset);
    const char* align_desc = (offset == 0)    ? "64-byte"
                             : (offset == 32) ? "32-byte"
                             : (offset == 16) ? "16-byte"
                             : (offset == 8)  ? "8-byte"
                             : (offset == 4)  ? "4-byte"
                                              : "unaligned";
    std::cout << std::setw(12) << offset << std::setw(15) << std::fixed << std::setprecision(2) << time << std::setw(15) << align_desc << "\n";
  }
}
