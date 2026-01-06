/**
 * @file common.h
 * @brief Common utilities and base classes for I/O library
 */
#pragma once

#include <cstdio>
#include <cstdlib>

#define ASSERT(exp)                                                             \
  do {                                                                          \
    if (!(exp)) {                                                               \
      fprintf(stderr, "[%s:%d] %s assertion fail\n", __FILE__, __LINE__, #exp); \
      abort();                                                                  \
    }                                                                           \
  } while (0)

struct NoCopy {
 protected:
  NoCopy() = default;
  ~NoCopy() = default;
  NoCopy(NoCopy&&) = default;
  NoCopy& operator=(NoCopy&&) = default;
  NoCopy(const NoCopy&) = delete;
  NoCopy& operator=(const NoCopy&) = delete;
};
