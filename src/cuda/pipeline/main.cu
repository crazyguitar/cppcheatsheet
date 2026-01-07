// Pipeline Basic - Double buffering with cuda::pipeline
//
// Demonstrates overlapping memory copies with compute using a 2-stage pipeline.
// While one buffer is being processed, the next tile is being loaded.
//
// Pipeline Flow (2 stages, 4 tiles):
// ==================================
//
//   Time -->
//   Stage 0: [Load T0]         [Load T2]         [     ]
//   Stage 1:          [Load T1]         [Load T3]
//   Compute:          [Proc T0][Proc T1][Proc T2][Proc T3]
//
//   Without pipeline (sequential):
//   [Load T0][Proc T0][Load T1][Proc T1][Load T2][Proc T2][Load T3][Proc T3]
//
// Pipeline API:
// =============
//
//   producer_acquire() --> memcpy_async() --> producer_commit()
//                              |
//                              v (async transfer in flight)
//                              |
//   consumer_wait() <----------+
//        |
//        v
//   [use data]
//        |
//        v
//   consumer_release() --> stage available for reuse

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cuda/pipeline>

__global__ void double_buffer(const float* in, float* out, int n) {
  constexpr int TILE = 256;
  __shared__ float buf[2][TILE];

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  int tid = threadIdx.x;
  int num_tiles = (n + TILE - 1) / TILE;

  // Prefetch first tile into stage 0
  pipe.producer_acquire();
  if (tid < TILE && tid < n) cuda::memcpy_async(&buf[0][tid], &in[tid], sizeof(float), pipe);
  pipe.producer_commit();

  for (int t = 0; t < num_tiles; t++) {
    int curr = t % 2;
    int next = (t + 1) % 2;

    // Prefetch next tile (if exists)
    if (t + 1 < num_tiles) {
      pipe.producer_acquire();
      int idx = (t + 1) * TILE + tid;
      if (tid < TILE && idx < n) cuda::memcpy_async(&buf[next][tid], &in[idx], sizeof(float), pipe);
      pipe.producer_commit();
    }

    // Wait for current tile
    pipe.consumer_wait();
    __syncthreads();

    // Process: double each value
    int idx = t * TILE + tid;
    if (tid < TILE && idx < n) out[idx] = buf[curr][tid] * 2.0f;

    pipe.consumer_release();
    __syncthreads();
  }
}

TEST(CUDA, PipelineDoubleBuffer) {
  constexpr int N = 1024;
  constexpr size_t size = N * sizeof(float);

  float *h_in, *h_out;
  float *d_in, *d_out;

  cudaMallocHost(&h_in, size);
  cudaMallocHost(&h_out, size);
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);

  for (int i = 0; i < N; i++) h_in[i] = static_cast<float>(i);
  cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

  double_buffer<<<1, 256>>>(d_in, d_out, N);

  cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(h_out[i], h_in[i] * 2.0f);
  }

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFreeHost(h_in);
  cudaFreeHost(h_out);
}
