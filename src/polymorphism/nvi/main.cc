#include <gtest/gtest.h>

#include <string>
#include <string_view>

namespace nvi {

struct Logger {
  std::string log(std::string_view msg) {
    std::string out;
    out += prefix();
    out += write(msg);
    out += suffix();
    return out;
  }
  virtual ~Logger() = default;

 private:
  std::string prefix() const { return "["; }
  std::string suffix() const { return "]"; }
  virtual std::string write(std::string_view msg) = 0;
};

struct UpperLogger : Logger {
 private:
  std::string write(std::string_view msg) override {
    std::string s(msg);
    for (auto& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
  }
};

struct PassthroughLogger : Logger {
 private:
  std::string write(std::string_view msg) override { return std::string(msg); }
};

}  // namespace nvi

TEST(NVI, BaseControlsFramingDerivedCustomizesBody) {
  nvi::UpperLogger u;
  nvi::PassthroughLogger p;
  EXPECT_EQ(u.log("hi"), "[HI]");
  EXPECT_EQ(p.log("hi"), "[hi]");
}
