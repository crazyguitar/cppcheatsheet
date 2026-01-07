#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

TEST(ReadLines, Getline) {
  char tmpl[] = "/tmp/testXXXXXX";
  int fd = mkstemp(tmpl);
  FILE* f = fdopen(fd, "w+");
  fprintf(f, "line1\nline2\n");
  rewind(f);

  char* line = nullptr;
  size_t len = 0;
  while (getline(&line, &len, f) != -1) {
  }

  free(line);
  fclose(f);
  unlink(tmpl);
}
