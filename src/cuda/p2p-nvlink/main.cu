/**
 * P2P Direct Access Ring Communication (Single-Process)
 *
 * Single process owns all GPUs. cudaDeviceEnablePeerAccess() creates a mapping
 * in GPU's page table allowing direct load/store to peer GPU memory.
 *
 * ┌─────────────────────────────────────────┐
 * │ Process                                 │
 * │ ┌─────────┐         ┌─────────┐         │
 * │ │ GPU 0   │ ══════▶ │ GPU 1   │         │  ✓ Same address space
 * │ │ ptr=A   │         │ ptr=B   │         │  ✓ Can dereference B from GPU 0
 * │ └─────────┘         └─────────┘         │
 * └─────────────────────────────────────────┘
 *
 * Memory Access Path (GPU 0 writing to GPU 1):
 *
 *   GPU 0 Thread              GPU 1 HBM
 *       │                         │
 *       │ st.global [peer], val   │
 *       ▼                         │
 *   ┌─────────┐                   │
 *   │ L2 Cache│ (miss)            │
 *   └────┬────┘                   │
 *        │      NVLink/PCIe       │
 *        └────────────────────────┼──▶ Written to GPU 1 HBM
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err); \
  } while (0)

__global__ void write_to_peer(int* __restrict__ peer_buf, int gpu_id, size_t len) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) peer_buf[idx] = gpu_id * 1000 + idx;
}

__global__ void verify_buffer(int* __restrict__ buf, int expected_gpu, size_t len, int* errors) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    int expected = expected_gpu * 1000 + idx;
    if (buf[idx] != expected) atomicAdd(errors, 1);
  }
}

/**
 * GPU 0                        GPU 1                        GPU N-1
 *   |                            |                            |
 *   |-- cudaMalloc(d_buf) -------|-- cudaMalloc(d_buf) -------|
 *   |                            |                            |
 *   |-- cudaDeviceEnablePeerAccess(all) ----------------------|
 *   |                            |                            |
 *   |-- write_to_peer(gpu1_buf) >|  (GPU 0 writes to GPU 1)   |
 *   |                            |-- write_to_peer(gpu2_buf) >|
 *   |  (GPU N-1 writes to GPU 0) <----------------------------|
 *   |                            |                            |
 *   |-- cudaDeviceSynchronize ---|-- cudaDeviceSynchronize ---|
 *   |                            |                            |
 *   |-- verify_buffer(d_buf) ----|-- verify_buffer(d_buf) ----|
 */
TEST(P2PNvlink, RingCommunication) {
  int num_gpus;
  CUDA_CHECK(cudaGetDeviceCount(&num_gpus));

  if (num_gpus < 2) {
    GTEST_SKIP() << "Need at least 2 GPUs for P2P test";
  }

  printf("Found %d GPUs\n", num_gpus);

  constexpr size_t kBufsize = 1024;
  std::vector<int*> bufs(num_gpus);
  std::vector<int*> d_errs(num_gpus);

  // Allocate buffers on each GPU
  for (int i = 0; i < num_gpus; ++i) {
    CUDA_CHECK(cudaSetDevice(i));
    CUDA_CHECK(cudaMalloc(&bufs[i], sizeof(int) * kBufsize));
    CUDA_CHECK(cudaMemset(bufs[i], 0, kBufsize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_errs[i], sizeof(int)));
    CUDA_CHECK(cudaMemset(d_errs[i], 0, sizeof(int)));
  }

  // Enable peer access between all GPU pairs
  for (int i = 0; i < num_gpus; ++i) {
    CUDA_CHECK(cudaSetDevice(i));
    for (int j = 0; j < num_gpus; ++j) {
      if (i != j) {
        int canAccess;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess, i, j));
        if (canAccess) {
          CUDA_CHECK(cudaDeviceEnablePeerAccess(j, 0));
        } else {
          printf("Warning: GPU %d cannot access GPU %d\n", i, j);
        }
      }
    }
  }

  // Print P2P attributes
  printf("\n=== P2P Attributes ===\n");
  for (int i = 0; i < num_gpus; ++i) {
    for (int j = 0; j < num_gpus; ++j) {
      if (i != j) {
        int atomic;
        CUDA_CHECK(cudaDeviceGetP2PAttribute(&atomic, cudaDevP2PAttrNativeAtomicSupported, i, j));
        printf("GPU %d -> GPU %d: %s\n", i, j, atomic ? "NVLink" : "PCIe");
      }
    }
  }

  int block = 256;
  int grid = (kBufsize + block - 1) / block;

  // Ring write: GPU i writes to GPU (i+1) % N
  for (int i = 0; i < num_gpus; ++i) {
    CUDA_CHECK(cudaSetDevice(i));
    int target = (i + 1) % num_gpus;
    write_to_peer<<<grid, block>>>(bufs[target], i, kBufsize);
  }

  // Sync all
  for (int i = 0; i < num_gpus; ++i) {
    CUDA_CHECK(cudaSetDevice(i));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Verify: GPU i should have data from GPU (i-1)
  for (int i = 0; i < num_gpus; ++i) {
    CUDA_CHECK(cudaSetDevice(i));
    int source = (i - 1 + num_gpus) % num_gpus;
    verify_buffer<<<grid, block>>>(bufs[i], source, kBufsize, d_errs[i]);
  }

  // Check results
  bool all_passed = true;
  for (int i = 0; i < num_gpus; ++i) {
    CUDA_CHECK(cudaSetDevice(i));
    CUDA_CHECK(cudaDeviceSynchronize());
    int source = (i - 1 + num_gpus) % num_gpus;
    int err = 0;
    CUDA_CHECK(cudaMemcpy(&err, d_errs[i], sizeof(int), cudaMemcpyDeviceToHost));
    printf("[GPU %d] Written by GPU %d, errors: %d %s\n", i, source, err, err == 0 ? "OK" : "FAIL");
    if (err != 0) all_passed = false;
  }

  // Cleanup
  for (int i = 0; i < num_gpus; ++i) {
    CUDA_CHECK(cudaSetDevice(i));
    for (int j = 0; j < num_gpus; ++j) {
      if (i != j) cudaDeviceDisablePeerAccess(j);
    }
    CUDA_CHECK(cudaFree(bufs[i]));
    CUDA_CHECK(cudaFree(d_errs[i]));
  }

  EXPECT_TRUE(all_passed);
}
