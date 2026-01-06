/**
 * @file handle.cc
 * @brief Handle implementation
 */

#include "handle.h"

#include "io.h"

void Handle::schedule() noexcept {
  if (state_ == Handle::kUnschedule) {
    IO::Get().Call(*this);
  }
}

void Handle::cancel() noexcept {
  if (state_ != Handle::kUnschedule) {
    IO::Get().Cancel(*this);
  }
}
