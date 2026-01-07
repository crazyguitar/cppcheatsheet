/**
 * CUDA IPC Demo (Single-Process Simulation)
 *
 * IPC (Inter-Process Communication) allows sharing GPU memory between processes.
 * This demo shows the IPC API usage in a single-process context using threads
 * to simulate multi-process behavior.
 *
 * Real IPC usage (with MPI):
 * ┌───────────────────┐   ┌───────────────────┐
 * │ Process 0 (Rank 0)│   │ Process 1 (Rank 1)│
 * │ ┌─────────┐       │   │  ┌─────────┐      │
 * │ │ GPU 0   │ ─ ─ ─ │─ ─│─▶│ GPU 1   │      │
 * │ └─────────┘       │   │  └─────────┘      │
 * └───────────────────┘   └───────────────────┘
 *   cudaIpcGetMemHandle     cudaIpcOpenMemHandle
 *
 * IPC Flow:
 *   1. Process A: cudaMalloc → cudaIpcGetMemHandle → send handle to Process B
 *   2. Process B: receive handle → cudaIpcOpenMemHandle → use pointer
 *   3. Process B: cudaIpcCloseMemHandle when done
 *
 * This example uses threads to demonstrate the API without requiring MPI.
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err); \
  } while (0)

__global__ void write_data(int* buf, int value, size_t len) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) buf[idx] = value * 1000 + idx;
}

__global__ void verify_data(int* buf, int expected_value, size_t len, int* errors) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    int expected = expected_value * 1000 + idx;
    if (buf[idx] != expected) atomicAdd(errors, 1);
  }
}

/**
 * IPC Handle Exchange Pattern:
 *
 *   Thread 0 (GPU 0)              Thread 1 (GPU 1)
 *       │                              │
 *       │── cudaMalloc ────────────────│
 *       │                              │
 *       │── cudaIpcGetMemHandle ───────│
 *       │                              │
 *       │══════ share handle ═════════▶│
 *       │                              │
 *       │                              │── cudaIpcOpenMemHandle
 *       │                              │
 *       │                              │── write_data (to GPU 0's buf)
 *       │                              │
 *       │◀═════════ barrier ══════════▶│
 *       │                              │
 *       │── verify_data ───────────────│
 *       │                              │
 *       │                              │── cudaIpcCloseMemHandle
 */
TEST(P2PIPC, ThreadSimulation) {
  int num_gpus;
  CUDA_CHECK(cudaGetDeviceCount(&num_gpus));

  if (num_gpus < 2) {
    GTEST_SKIP() << "Need at least 2 GPUs for IPC test";
  }

  printf("IPC Demo with %d GPUs (using threads to simulate processes)\n", num_gpus);

  constexpr size_t kBufsize = 1024;

  // Shared state between "processes" (threads)
  std::vector<cudaIpcMemHandle_t> handles(num_gpus);
  std::vector<int*> local_bufs(num_gpus);
  std::atomic<int> ready_count{0};
  std::atomic<int> done_count{0};
  std::vector<int> errors(num_gpus, 0);

  auto worker = [&](int gpu_id) {
    cudaSetDevice(gpu_id);

    // Allocate local buffer
    int* d_buf;
    cudaMalloc(&d_buf, sizeof(int) * kBufsize);
    cudaMemset(d_buf, 0, kBufsize * sizeof(int));
    local_bufs[gpu_id] = d_buf;

    // Get IPC handle for our buffer
    cudaIpcGetMemHandle(&handles[gpu_id], d_buf);

    // Signal ready and wait for all
    ready_count++;
    while (ready_count < num_gpus) std::this_thread::yield();

    // Open handle to next GPU's buffer (ring pattern)
    int target = (gpu_id + 1) % num_gpus;
    int* peer_buf;
    cudaIpcOpenMemHandle((void**)&peer_buf, handles[target], cudaIpcMemLazyEnablePeerAccess);

    // Write to peer's buffer
    int block = 256;
    int grid = (kBufsize + block - 1) / block;
    write_data<<<grid, block>>>(peer_buf, gpu_id, kBufsize);
    cudaDeviceSynchronize();

    // Close peer handle
    cudaIpcCloseMemHandle(peer_buf);

    // Signal done writing and wait for all
    done_count++;
    while (done_count < num_gpus) std::this_thread::yield();

    // Verify our buffer was written by previous GPU
    int source = (gpu_id - 1 + num_gpus) % num_gpus;
    int* d_err;
    cudaMalloc(&d_err, sizeof(int));
    cudaMemset(d_err, 0, sizeof(int));
    verify_data<<<grid, block>>>(d_buf, source, kBufsize, d_err);
    cudaMemcpy(&errors[gpu_id], d_err, sizeof(int), cudaMemcpyDeviceToHost);

    printf("[GPU %d] Written by GPU %d, errors: %d %s\n", gpu_id, source, errors[gpu_id], errors[gpu_id] == 0 ? "OK" : "FAIL");

    cudaFree(d_err);
    cudaFree(d_buf);
  };

  // Launch "processes" as threads
  std::vector<std::thread> threads;
  for (int i = 0; i < num_gpus; ++i) {
    threads.emplace_back(worker, i);
  }

  for (auto& t : threads) {
    t.join();
  }

  // Check all passed
  bool all_passed = true;
  for (int i = 0; i < num_gpus; ++i) {
    if (errors[i] != 0) all_passed = false;
  }

  EXPECT_TRUE(all_passed);
}

// Basic IPC API demonstration
TEST(P2PIPC, BasicAPI) {
  int num_gpus;
  CUDA_CHECK(cudaGetDeviceCount(&num_gpus));

  if (num_gpus < 1) {
    GTEST_SKIP() << "No GPU available";
  }

  printf("\n=== IPC API Demo ===\n");

  // Allocate buffer
  int* d_buf;
  CUDA_CHECK(cudaMalloc(&d_buf, sizeof(int) * 1024));
  printf("Allocated buffer at %p\n", d_buf);

  // Get IPC handle
  cudaIpcMemHandle_t handle;
  CUDA_CHECK(cudaIpcGetMemHandle(&handle, d_buf));
  printf("Got IPC handle (64 bytes, can be sent to other process)\n");

  // In real multi-process usage:
  // - Send 'handle' to another process via pipe/socket/MPI
  // - Other process calls cudaIpcOpenMemHandle to get usable pointer
  // - Other process can then read/write the memory
  // - Other process calls cudaIpcCloseMemHandle when done

  printf("\nIPC handle can be shared via:\n");
  printf("  - MPI_Send/MPI_Recv\n");
  printf("  - Unix domain socket\n");
  printf("  - Shared memory file\n");
  printf("  - Any IPC mechanism that can transfer 64 bytes\n");

  CUDA_CHECK(cudaFree(d_buf));
  EXPECT_TRUE(true);
}
