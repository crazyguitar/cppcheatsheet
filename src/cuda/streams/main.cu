// CUDA Streams - Overlapping transfers and computation
//
// This example demonstrates how to use multiple CUDA streams to overlap
// data transfers with kernel execution. By dividing work into chunks and
// processing each chunk in a separate stream, we can hide transfer latency.
//
// Key concepts:
// - Streams are independent execution queues on the GPU
// - Operations in the same stream execute in order
// - Operations in different streams may execute concurrently
// - cudaMallocHost() (pinned memory) is REQUIRED for async transfers
// - cudaMemcpyAsync() returns immediately, transfer happens in background
//
// Pitfalls:
// - Using regular malloc() instead of cudaMallocHost() makes async transfers
//   synchronous, eliminating all overlap benefits
// - Forgetting to synchronize before reading results causes data races
// - Too many streams can cause overhead; 2-4 streams is often optimal
// - Stream synchronization order matters for correctness
//
// Timeline with 4 streams (H2D = Host to Device, D2H = Device to Host):
// Stream 0: [H2D][Kernel][D2H]
// Stream 1:      [H2D][Kernel][D2H]
// Stream 2:           [H2D][Kernel][D2H]
// Stream 3:                [H2D][Kernel][D2H]

#include <cuda_runtime.h>
#include <gtest/gtest.h>

__global__ void kernel(float* data, int n, int offset) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    data[i] = data[i] * 2.0f + static_cast<float>(offset);
  }
}

struct StreamsTest {
  static constexpr int n = 1024 * 1024;
  static constexpr int num_streams = 4;
  static constexpr int chunk_size = n / num_streams;
  static constexpr size_t size = n * sizeof(float);
  static constexpr size_t chunk_bytes = chunk_size * sizeof(float);

  float* h_data;
  float* d_data;
  cudaStream_t streams[num_streams];

  StreamsTest() {
    cudaMallocHost(&h_data, size);
    cudaMalloc(&d_data, size);

    for (int i = 0; i < num_streams; i++) {
      cudaStreamCreate(&streams[i]);
    }

    for (int i = 0; i < n; i++) {
      h_data[i] = static_cast<float>(i);
    }
  }

  ~StreamsTest() {
    for (int i = 0; i < num_streams; i++) {
      cudaStreamDestroy(streams[i]);
    }
    cudaFreeHost(h_data);
    cudaFree(d_data);
  }

  void run() {
    for (int i = 0; i < num_streams; i++) {
      int offset = i * chunk_size;

      cudaMemcpyAsync(d_data + offset, h_data + offset, chunk_bytes, cudaMemcpyHostToDevice, streams[i]);

      int threads = 256;
      int blocks = (chunk_size + threads - 1) / threads;
      kernel<<<blocks, threads, 0, streams[i]>>>(d_data + offset, chunk_size, i);

      cudaMemcpyAsync(h_data + offset, d_data + offset, chunk_bytes, cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < num_streams; i++) {
      cudaStreamSynchronize(streams[i]);
    }
  }
};

TEST(CUDA, Streams) {
  StreamsTest test;

  // Save original values for verification
  float orig_0 = test.h_data[0];
  float orig_chunk = test.h_data[StreamsTest::chunk_size];

  test.run();

  // Stream 0: data[i] = data[i] * 2 + 0
  EXPECT_FLOAT_EQ(test.h_data[0], orig_0 * 2.0f + 0.0f);

  // Stream 1: data[i] = data[i] * 2 + 1
  EXPECT_FLOAT_EQ(test.h_data[StreamsTest::chunk_size], orig_chunk * 2.0f + 1.0f);
}
