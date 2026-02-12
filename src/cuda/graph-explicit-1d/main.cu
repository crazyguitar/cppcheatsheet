// CUDA Graph - Explicit API with cudaGraphAddMemcpyNode1D
//
// This example is a simplified version of graph-explicit that uses
// cudaGraphAddMemcpyNode1D (CUDA 11.1+) instead of cudaMemcpy3DParms.
// For flat 1D copies, this avoids the verbose cudaPitchedPtr setup.
//
// Key concepts:
// - cudaGraphAddMemcpyNode1D() wraps a 1D memcpy as a graph node
// - Same dependency semantics as cudaGraphAddMemcpyNode()
// - Much simpler for linear buffers (no pitch/extent needed)
//
// Compare with graph-explicit/ which uses cudaMemcpy3DParms for the
// same H2D copy — useful when working with 2D/3D pitched memory.
//
// Graph structure:
//   [memcpy H2D (1D)]
//         |
//     [scale ×2]
//       /    \
//  [add +10] [add +100]   ← parallel branches

#include <cuda_runtime.h>
#include <gtest/gtest.h>

__global__ void scale_k(float* data, int n, float factor) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] *= factor;
}

__global__ void add_k(float* out, const float* in, int n, float bias) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] + bias;
}

TEST(CUDA, GraphExplicitAPI1D) {
  constexpr int n = 1024;
  constexpr size_t size = n * sizeof(float);
  constexpr int threads = 256;
  constexpr int blocks = (n + threads - 1) / threads;

  float *h_data, *d_in, *d_outB, *d_outC;
  cudaMallocHost(&h_data, size);
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_outB, size);
  cudaMalloc(&d_outC, size);

  for (int i = 0; i < n; i++) h_data[i] = static_cast<float>(i);

  cudaGraph_t graph;
  cudaGraphCreate(&graph, 0);

  // Node 0: memcpy H2D using 1D API (no cudaPitchedPtr needed)
  cudaGraphNode_t copyH2D;
  cudaGraphAddMemcpyNode1D(&copyH2D, graph, nullptr, 0, d_in, h_data, size, cudaMemcpyHostToDevice);

  // Node 1: scale depends on H2D
  cudaGraphNode_t nodeA;
  float factor = 2.0f;
  void* argsA[] = {&d_in, (void*)&n, &factor};
  cudaKernelNodeParams paramsA = {};
  paramsA.func = (void*)scale_k;
  paramsA.gridDim = dim3(blocks);
  paramsA.blockDim = dim3(threads);
  paramsA.kernelParams = argsA;
  cudaGraphAddKernelNode(&nodeA, graph, &copyH2D, 1, &paramsA);

  // Node 2 & 3: parallel branches depend on A
  cudaGraphNode_t nodeB;
  float biasB = 10.0f;
  void* argsB[] = {&d_outB, &d_in, (void*)&n, &biasB};
  cudaKernelNodeParams paramsB = {};
  paramsB.func = (void*)add_k;
  paramsB.gridDim = dim3(blocks);
  paramsB.blockDim = dim3(threads);
  paramsB.kernelParams = argsB;
  cudaGraphAddKernelNode(&nodeB, graph, &nodeA, 1, &paramsB);

  cudaGraphNode_t nodeC;
  float biasC = 100.0f;
  void* argsC[] = {&d_outC, &d_in, (void*)&n, &biasC};
  cudaKernelNodeParams paramsC = {};
  paramsC.func = (void*)add_k;
  paramsC.gridDim = dim3(blocks);
  paramsC.blockDim = dim3(threads);
  paramsC.kernelParams = argsC;
  cudaGraphAddKernelNode(&nodeC, graph, &nodeA, 1, &paramsC);

  cudaGraphExec_t instance;
  cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaGraphLaunch(instance, stream);
  cudaStreamSynchronize(stream);

  float* h_outB = new float[n];
  float* h_outC = new float[n];
  cudaMemcpy(h_outB, d_outB, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_outC, d_outC, size, cudaMemcpyDeviceToHost);

  EXPECT_FLOAT_EQ(h_outB[0], 10.0f);   // 0 * 2 + 10
  EXPECT_FLOAT_EQ(h_outB[5], 20.0f);   // 5 * 2 + 10
  EXPECT_FLOAT_EQ(h_outC[0], 100.0f);  // 0 * 2 + 100
  EXPECT_FLOAT_EQ(h_outC[5], 110.0f);  // 5 * 2 + 100

  delete[] h_outB;
  delete[] h_outC;
  cudaGraphExecDestroy(instance);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  cudaFreeHost(h_data);
  cudaFree(d_in);
  cudaFree(d_outB);
  cudaFree(d_outC);
}
