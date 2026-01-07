/**
 * @file progress.h
 * @brief Progress tracking and bandwidth reporting for I/O operations
 */
#pragma once
#include <chrono>
#include <cstdio>
#include <iostream>

#include "common.h"

/**
 * @brief Progress tracker for monitoring bandwidth and operation throughput
 *
 * Tracks elapsed time, completed operations, and bandwidth utilization.
 * Displays real-time progress with bandwidth in Gbps and percentage of total capacity.
 */
class Progress : private NoCopy {
 public:
  using nanoseconds = std::chrono::nanoseconds;
  using seconds = std::chrono::seconds;
  using timepoint = std::chrono::high_resolution_clock::time_point;

  /** @brief Conversion factor: bytes to gigabits */
  constexpr static double Gb = 8.0f / 1e9;
  /** @brief Print frequency to reduce overhead */
  constexpr static int kPrintFreq = 64;

  Progress() = default;
  /**
   * @brief Construct progress tracker with expected totals
   * @param total_ops Expected total number of operations
   * @param total_bw Expected total bandwidth in bits/sec
   * @param name Optional test name to display
   */
  Progress(size_t total_ops, size_t total_bw, std::string_view name = "") : total_ops_{total_ops}, total_bw_{total_bw}, name_{name} {}
  Progress(Progress&& other) = delete;
  Progress& operator=(Progress&& other) = delete;

  /**
   * @brief Print current progress to stdout
   * @param now Current timestamp
   * @param size Size per operation in bytes
   * @param ops Number of operations completed
   */
  void Print(timepoint now, size_t size, uint64_t ops) { PrintProgress(name_, start_, now, size, ops, total_ops_, total_bw_); }

 private:
  // clang-format off
  /**
   * @brief Print formatted progress line with bandwidth statistics
   * @param name Test name to display
   * @param start Start timestamp
   * @param end Current timestamp
   * @param size Size per operation in bytes
   * @param ops Number of operations completed
   * @param total_ops Expected total operations
   * @param total_bw Expected total bandwidth in bits/sec
   *
   * Output format: [name] [time] ops=current/total bytes=current/total bw=X.XXXGbps(XX.X%) lat=X.XXXus
   */
  static void PrintProgress(
    std::string_view name,
    timepoint start,
    timepoint end,
    size_t size,
    uint64_t ops,
    uint64_t total_ops,
    size_t total_bw
  ) {
    auto elapse = std::chrono::duration_cast<nanoseconds>(end - start).count() / 1e9;
    auto bytes = size * ops;
    auto total_bytes = size * total_ops;
    auto bw_gbps = bytes * Gb / elapse;
    auto total_bw_gbs = total_bw * 1e-9;
    auto percent = 100.0 * bw_gbps / (total_bw_gbs);
    auto lat_us = (elapse * 1e6) / ops;
    if (name.empty()) {
      std::cout << fmt::format("\r[{:.3f}s] ops={}/{} bytes={}/{} bw={:.3f}Gbps({:.1f}%) lat={:.3f}us\033[K", elapse, ops, total_ops, bytes, total_bytes, bw_gbps, percent, lat_us) << std::flush;
    } else {
      std::cout << fmt::format("\r[{}] [{:.3f}s] ops={}/{} bytes={}/{} bw={:.3f}Gbps({:.1f}%) lat={:.3f}us\033[K", name, elapse, ops, total_ops, bytes, total_bytes, bw_gbps, percent, lat_us) << std::flush;
    }
  }
  // clang-format on

 private:
  size_t total_ops_ = 0;
  size_t total_bw_ = 0;
  std::string_view name_;
  timepoint start_{std::chrono::high_resolution_clock::now()};
};
