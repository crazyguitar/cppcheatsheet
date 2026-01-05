#include <iostream>
#include <string>
#include <type_traits>

template <typename S, typename = std::enable_if_t<std::is_same_v<std::string, std::decay_t<S>>>>
void Foo(S s) {
  std::cout << s << "\n";
}

template <typename T>
void Bar(T) {}
template <>
void Bar<int>(int s) {
  std::cout << "int: " << s << "\n";
}
template <>
void Bar<std::string>(std::string s) {
  std::cout << "str: " << s << "\n";
}

int main() {
  Foo(std::string("hello"));
  Bar(123);
  Bar(std::string("world"));
}
