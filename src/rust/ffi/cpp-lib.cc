// C++ library to be called from Rust
#include <cmath>
#include <cstdint>
#include <cstring>

extern "C" {
// Simple function: add two integers
int32_t cpp_add(int32_t a, int32_t b) { return a + b; }

// Function with pointer parameter
void cpp_fill_array(int32_t* arr, size_t len, int32_t value) {
  for (size_t i = 0; i < len; ++i) {
    arr[i] = value;
  }
}

// Function returning heap-allocated string (caller must free)
char* cpp_create_greeting(const char* name) {
  const char* prefix = "Hello, ";
  const char* suffix = "!";
  size_t len = strlen(prefix) + strlen(name) + strlen(suffix) + 1;
  char* result = new char[len];
  strcpy(result, prefix);
  strcat(result, name);
  strcat(result, suffix);
  return result;
}

// Free string allocated by C++
void cpp_free_string(char* s) { delete[] s; }

// Struct passed between Rust and C++
struct Point {
  double x;
  double y;
};

double cpp_distance(const Point* p1, const Point* p2) {
  double dx = p2->x - p1->x;
  double dy = p2->y - p1->y;
  return sqrt(dx * dx + dy * dy);
}

Point cpp_midpoint(const Point* p1, const Point* p2) { return Point{(p1->x + p2->x) / 2.0, (p1->y + p2->y) / 2.0}; }
}
