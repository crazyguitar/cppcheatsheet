// CUDA Graph - Memory Nodes (CUDA 11.4+)
//
// This example demonstrates cudaGraphAddMemAllocNode and
// cudaGraphAddMemFreeNode to manage memory allocation as part of the graph.
// The driver can optimize memory reuse across repeated graph launches.
//
// Key concepts:
// - cudaGraphAddMemAllocNode() allocates device memory within the graph
// - The allocated pointer (dptr) is valid after instantiation
// - cudaGraphAddMemFreeNode() frees memory allocated by an alloc node
// - Memory lifetime is tied to the graph execution
//
// Pitfalls:
// - Requires CUDA 11.4+ and a device with compute capability 7.0+
// - The dptr is only valid between the alloc and free nodes in the graph
// - Must check device supports memory pools (cudaDevAttrMemoryPoolsSupported)
//
// Graph structure:
//   [memAlloc d_tmp]
//         |
//    [fill d_tmp=42]
//         |
//   [copy_add d_out = d_tmp + 8]
//         |
//   [memFree d_tmp]

#include <cuda_runtime.h>
#include <gtest/gtest.h>

__global__ void fill(float* data, int n, float val) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] = val;
}

__global__ void copy_add(float* dst, const float* src, int n, float bias) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i] + bias;
}

TEST(CUDA, GraphMemoryNodes) {
  // Check memory pool support
  int dev = 0;
  int poolSupport = 0;
  cudaDeviceGetAttribute(&poolSupport, cudaDevAttrMemoryPoolsSupported, dev);
  if (!poolSupport) {
    GTEST_SKIP() << "Device does not support memory pools";
  }

  constexpr int n = 1024;
  constexpr size_t size = n * sizeof(float);
  constexpr int threads = 256;
  constexpr int blocks = (n + threads - 1) / threads;

  float* d_out;
  cudaMalloc(&d_out, size);

  cudaGraph_t graph;
  cudaGraphCreate(&graph, 0);

  // Alloc node: allocate temporary buffer inside the graph
  cudaMemAllocNodeParams allocParams = {};
  allocParams.poolProps.allocType = cudaMemAllocationTypePinned;
  allocParams.poolProps.location.type = cudaMemLocationTypeDevice;
  allocParams.poolProps.location.id = dev;
  allocParams.bytesize = size;

  cudaGraphNode_t allocNode;
  cudaGraphAddMemAllocNode(&allocNode, graph, nullptr, 0, &allocParams);
  float* d_tmp = (float*)allocParams.dptr;

  // Fill temp buffer (depends on alloc)
  cudaGraphNode_t fillNode;
  float val = 42.0f;
  void* fillArgs[] = {&d_tmp, (void*)&n, &val};
  cudaKernelNodeParams fillParams = {};
  fillParams.func = (void*)fill;
  fillParams.gridDim = dim3(blocks);
  fillParams.blockDim = dim3(threads);
  fillParams.kernelParams = fillArgs;
  cudaGraphAddKernelNode(&fillNode, graph, &allocNode, 1, &fillParams);

  // Copy from temp to output with bias (depends on fill)
  cudaGraphNode_t copyNode;
  float bias = 8.0f;
  void* copyArgs[] = {&d_out, &d_tmp, (void*)&n, &bias};
  cudaKernelNodeParams copyParams = {};
  copyParams.func = (void*)copy_add;
  copyParams.gridDim = dim3(blocks);
  copyParams.blockDim = dim3(threads);
  copyParams.kernelParams = copyArgs;
  cudaGraphAddKernelNode(&copyNode, graph, &fillNode, 1, &copyParams);

  // Free temp buffer (depends on copy)
  cudaGraphNode_t freeNode;
  cudaGraphAddMemFreeNode(&freeNode, graph, &copyNode, 1, d_tmp);

  // Instantiate and launch
  cudaGraphExec_t instance;
  cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaGraphLaunch(instance, stream);
  cudaStreamSynchronize(stream);

  float* h_out = new float[n];
  cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
  EXPECT_FLOAT_EQ(h_out[0], 50.0f);  // 42 + 8
  EXPECT_FLOAT_EQ(h_out[100], 50.0f);

  delete[] h_out;
  cudaGraphExecDestroy(instance);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  cudaFree(d_out);
}
