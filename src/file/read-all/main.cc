#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

TEST(ReadAll, IntoMemory) {
  char tmpl[] = "/tmp/testXXXXXX";
  int fd = mkstemp(tmpl);
  FILE* f = fdopen(fd, "w+");
  fprintf(f, "test content");
  rewind(f);

  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  rewind(f);

  char* buf = (char*)malloc(size + 1);
  fread(buf, 1, size, f);
  buf[size] = '\0';

  free(buf);
  fclose(f);
  unlink(tmpl);
}
