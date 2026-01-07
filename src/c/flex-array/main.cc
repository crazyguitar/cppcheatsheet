#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>

struct Message {
  int length;
  char data[];
};

TEST(FlexArray, VariableSizeStruct) {
  const char* text = "Hello";
  int len = strlen(text);

  Message* msg = (Message*)malloc(sizeof(Message) + len + 1);
  msg->length = len;
  strcpy(msg->data, text);

  EXPECT_EQ(msg->length, 5);
  EXPECT_STREQ(msg->data, "Hello");

  free(msg);
}
