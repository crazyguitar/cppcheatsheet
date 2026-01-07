#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>

namespace {
char** split(char* str, char delim) {
  int count = 1;
  for (char* p = str; *p; p++) {
    if (*p == delim) count++;
  }
  char** result = (char**)calloc(count + 1, sizeof(char*));
  char delims[2] = {delim, '\0'};
  char* token = strtok(str, delims);
  int i = 0;
  while (token) {
    result[i++] = strdup(token);
    token = strtok(NULL, delims);
  }
  return result;
}

void free_tokens(char** tokens) {
  for (char** p = tokens; *p; p++) free(*p);
  free(tokens);
}
}  // namespace

TEST(SplitString, SplitsByDelimiter) {
  char str[] = "hello,world,test";
  char** tokens = split(str, ',');

  EXPECT_STREQ(tokens[0], "hello");
  EXPECT_STREQ(tokens[1], "world");
  EXPECT_STREQ(tokens[2], "test");
  EXPECT_EQ(tokens[3], nullptr);

  free_tokens(tokens);
}
