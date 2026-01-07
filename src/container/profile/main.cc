#include <chrono>
#include <deque>
#include <iostream>
#include <list>
#include <vector>

using milliseconds = std::chrono::milliseconds;

template <typename F>
void profile(const char* label, F&& func) {
  const auto start = std::chrono::steady_clock::now();
  func();
  const auto end = std::chrono::steady_clock::now();
  const auto ms = std::chrono::duration_cast<milliseconds>(end - start);
  std::cout << label << ": " << ms.count() << " ms\n";
}

constexpr int N = 100000;

int main() {
  std::cout << "=== push_back ===\n";
  profile("vector", [] {
    std::vector<int> v;
    for (int i = 0; i < N; ++i) v.push_back(i);
  });
  profile("deque", [] {
    std::deque<int> d;
    for (int i = 0; i < N; ++i) d.push_back(i);
  });
  profile("list", [] {
    std::list<int> l;
    for (int i = 0; i < N; ++i) l.push_back(i);
  });

  std::cout << "\n=== push_front ===\n";
  profile("vector", [] {
    std::vector<int> v;
    for (int i = 0; i < N / 10; ++i) v.insert(v.begin(), i);
  });
  profile("deque", [] {
    std::deque<int> d;
    for (int i = 0; i < N; ++i) d.push_front(i);
  });
  profile("list", [] {
    std::list<int> l;
    for (int i = 0; i < N; ++i) l.push_front(i);
  });

  std::cout << "\n=== pop_back ===\n";
  {
    std::vector<int> v(N);
    std::deque<int> d(N);
    std::list<int> l(N);
    profile("vector", [&] {
      while (!v.empty()) v.pop_back();
    });
    profile("deque", [&] {
      while (!d.empty()) d.pop_back();
    });
    profile("list", [&] {
      while (!l.empty()) l.pop_back();
    });
  }

  std::cout << "\n=== pop_front ===\n";
  {
    std::vector<int> v(N / 10);
    std::deque<int> d(N);
    std::list<int> l(N);
    profile("vector", [&] {
      while (!v.empty()) v.erase(v.begin());
    });
    profile("deque", [&] {
      while (!d.empty()) d.pop_front();
    });
    profile("list", [&] {
      while (!l.empty()) l.pop_front();
    });
  }
}
