#define _GNU_SOURCE
#include <ftw.h>
#include <gtest/gtest.h>

static int callback(const char* path, const struct stat* sb, int type, struct FTW* ftwbuf) { return 0; }

TEST(TreeWalk, Nftw) { nftw("/tmp", callback, 4, FTW_DEPTH | FTW_PHYS); }
