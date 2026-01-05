#include <gtest/gtest.h>

#include <string>

class Document {
 public:
  Document(const std::string& content) : content_(content) {}
  const std::string& content() const { return content_; }

 private:
  std::string content_;
};

TEST(RuleOfZero, DefaultCopyWorks) {
  Document doc1("Rule of Zero");
  Document doc2 = doc1;
  EXPECT_EQ(doc2.content(), "Rule of Zero");
}

TEST(RuleOfZero, DefaultMoveWorks) {
  Document doc1("Rule of Zero");
  Document doc2 = std::move(doc1);
  EXPECT_EQ(doc2.content(), "Rule of Zero");
}
