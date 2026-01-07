#include <gtest/gtest.h>

#include <sstream>
#include <string>

std::string log_output;

#define LOG(fmt, ...)                               \
  do {                                              \
    char buf[256];                                  \
    snprintf(buf, sizeof(buf), fmt, ##__VA_ARGS__); \
    log_output += buf;                              \
  } while (0)

TEST(VariadicMacros, HandlesVariableArgs) {
  log_output.clear();
  LOG("Hello");
  EXPECT_EQ(log_output, "Hello");

  log_output.clear();
  LOG("Value: %d", 42);
  EXPECT_EQ(log_output, "Value: 42");
}
