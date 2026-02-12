// CUDA Graph - Graph Update
//
// This example demonstrates updating a graph executable in-place without
// re-instantiation. cudaGraphExecKernelNodeSetParams() changes kernel
// parameters while keeping the same topology, avoiding the expensive
// instantiation step.
//
// Key concepts:
// - cudaGraphExecKernelNodeSetParams() updates a kernel node's parameters
// - cudaGraphExecUpdate() applies a modified graph to an existing executable
// - Update fails if the graph topology changes; must re-instantiate
// - In-place updates are much cheaper than destroy + re-instantiate
//
// Pitfalls:
// - The node handle must come from the original graph creation
// - Changing topology (adding/removing nodes) requires re-instantiation
// - Updated parameters take effect on the next cudaGraphLaunch()
//
// Graph structure:
//   [scale ×factor]   ← single node, factor updated in-place

#include <cuda_runtime.h>
#include <gtest/gtest.h>

__global__ void scale(float* data, int n, float factor) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] *= factor;
}

TEST(CUDA, GraphUpdate) {
  constexpr int n = 1024;
  constexpr size_t size = n * sizeof(float);
  constexpr int threads = 256;
  constexpr int blocks = (n + threads - 1) / threads;

  float *h_data, *d_data;
  cudaMallocHost(&h_data, size);
  cudaMalloc(&d_data, size);

  // Build graph with explicit API to get node handle
  cudaGraph_t graph;
  cudaGraphCreate(&graph, 0);

  cudaGraphNode_t node;
  float factor = 2.0f;
  void* args[] = {&d_data, (void*)&n, &factor};
  cudaKernelNodeParams params = {};
  params.func = (void*)scale;
  params.gridDim = dim3(blocks);
  params.blockDim = dim3(threads);
  params.kernelParams = args;
  cudaGraphAddKernelNode(&node, graph, nullptr, 0, &params);

  cudaGraphExec_t instance;
  cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Launch with factor=2
  for (int i = 0; i < n; i++) h_data[i] = 5.0f;
  cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
  cudaGraphLaunch(instance, stream);
  cudaStreamSynchronize(stream);
  cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
  EXPECT_FLOAT_EQ(h_data[0], 10.0f);  // 5 * 2

  // Update factor to 3 in-place (no re-instantiation)
  float newFactor = 3.0f;
  void* newArgs[] = {&d_data, (void*)&n, &newFactor};
  params.kernelParams = newArgs;
  cudaGraphExecKernelNodeSetParams(instance, node, &params);

  for (int i = 0; i < n; i++) h_data[i] = 5.0f;
  cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
  cudaGraphLaunch(instance, stream);
  cudaStreamSynchronize(stream);
  cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
  EXPECT_FLOAT_EQ(h_data[0], 15.0f);  // 5 * 3

  cudaGraphExecDestroy(instance);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  cudaFreeHost(h_data);
  cudaFree(d_data);
}
