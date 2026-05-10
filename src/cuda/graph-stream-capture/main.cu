// CUDA Graph - Stream Capture
//
// This example demonstrates creating a CUDA graph by capturing stream
// operations. Stream capture records kernel launches and memory copies into
// a graph without executing them. The graph is then instantiated once and
// replayed multiple times, eliminating per-launch CPU overhead.
//
// Key concepts:
// - cudaStreamBeginCapture() starts recording operations into a graph
// - cudaStreamEndCapture() finalizes the graph
// - cudaGraphInstantiate() compiles the graph into an executable
// - cudaGraphLaunch() replays the entire captured workflow
// - Graph instantiation is expensive; launch is cheap
//
// Pitfalls:
// - Operations during capture are NOT executed; don't read results before launch
// - cudaStreamCaptureModeGlobal blocks all non-captured streams during capture
// - Synchronization APIs (cudaDeviceSynchronize) inside capture cause errors
// - The captured stream must not be destroyed before ending capture
//
// Graph structure (captured from stream):
//   [memcpy H2D]
//        |
//    [scale ×2]
//        |
//   [add_bias +1]
//        |
//   [memcpy D2H]

#include <cuda_runtime.h>
#include <gtest/gtest.h>

__global__ void scale(float* data, int n, float factor) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] *= factor;
}

__global__ void add_bias(float* data, int n, float bias) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] += bias;
}

TEST(CUDA, GraphStreamCapture) {
  constexpr int n = 1024;
  constexpr size_t size = n * sizeof(float);

  float *h_data, *d_data;
  cudaMallocHost(&h_data, size);
  cudaMalloc(&d_data, size);

  for (int i = 0; i < n; i++) h_data[i] = static_cast<float>(i);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  // Capture: H2D -> scale -> add_bias -> D2H
  cudaGraph_t graph;
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

  cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);
  scale<<<blocks, threads, 0, stream>>>(d_data, n, 2.0f);
  add_bias<<<blocks, threads, 0, stream>>>(d_data, n, 1.0f);
  cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream);

  cudaStreamEndCapture(stream, &graph);

  // Instantiate once, launch many times
  cudaGraphExec_t instance;
  cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

  // First launch: data[i] = i * 2 + 1
  for (int i = 0; i < n; i++) h_data[i] = static_cast<float>(i);
  cudaGraphLaunch(instance, stream);
  cudaStreamSynchronize(stream);

  EXPECT_FLOAT_EQ(h_data[0], 1.0f);    // 0 * 2 + 1
  EXPECT_FLOAT_EQ(h_data[1], 3.0f);    // 1 * 2 + 1
  EXPECT_FLOAT_EQ(h_data[10], 21.0f);  // 10 * 2 + 1

  // Second launch reuses the same graph (no re-instantiation)
  for (int i = 0; i < n; i++) h_data[i] = 100.0f;
  cudaGraphLaunch(instance, stream);
  cudaStreamSynchronize(stream);

  EXPECT_FLOAT_EQ(h_data[0], 201.0f);  // 100 * 2 + 1

  cudaGraphExecDestroy(instance);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  cudaFreeHost(h_data);
  cudaFree(d_data);
}
