// CUDA Graph - Explicit API
//
// This example demonstrates building a CUDA graph using the explicit API
// where nodes and dependencies are added manually. This gives full control
// over the graph topology, enabling parallel branches that stream capture
// cannot easily express.
//
// Key concepts:
// - cudaGraphCreate() creates an empty graph
// - cudaGraphAddKernelNode() adds a kernel with explicit dependencies
// - Dependencies are specified as an array of predecessor nodes
// - Nodes with no dependency on each other can run in parallel
// - cudaGraphExecKernelNodeSetParams() updates parameters without re-instantiation
//
// Graph topology in this example:
//   memcpy H2D
//       |
//    [kernelA]  (scale by 2)
//      / \
// [kernelB] [kernelC]  (add 10, add 100, run in parallel)
//      \ /
//   memcpy D2H
//
// Pitfalls:
// - Kernel params struct must outlive the graph creation call
// - Forgetting dependencies causes data races between nodes
// - Graph update fails if topology changes; must re-instantiate

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

TEST(CUDA, GraphExplicitAPI) {
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

  // Node 0: memcpy H2D
  cudaGraphNode_t copyH2D;
  cudaMemcpy3DParms cpyParams = {};
  cpyParams.srcPtr = make_cudaPitchedPtr(h_data, size, n, 1);
  cpyParams.dstPtr = make_cudaPitchedPtr(d_in, size, n, 1);
  cpyParams.extent = make_cudaExtent(size, 1, 1);
  cpyParams.kind = cudaMemcpyHostToDevice;
  cudaGraphAddMemcpyNode(&copyH2D, graph, nullptr, 0, &cpyParams);

  // Node 1: kernelA (scale) depends on H2D
  cudaGraphNode_t nodeA;
  float factor = 2.0f;
  void* argsA[] = {&d_in, (void*)&n, &factor};
  cudaKernelNodeParams paramsA = {};
  paramsA.func = (void*)scale_k;
  paramsA.gridDim = dim3(blocks);
  paramsA.blockDim = dim3(threads);
  paramsA.kernelParams = argsA;
  cudaGraphAddKernelNode(&nodeA, graph, &copyH2D, 1, &paramsA);

  // Node 2: kernelB (add 10) depends on A
  cudaGraphNode_t nodeB;
  float biasB = 10.0f;
  void* argsB[] = {&d_outB, &d_in, (void*)&n, &biasB};
  cudaKernelNodeParams paramsB = {};
  paramsB.func = (void*)add_k;
  paramsB.gridDim = dim3(blocks);
  paramsB.blockDim = dim3(threads);
  paramsB.kernelParams = argsB;
  cudaGraphAddKernelNode(&nodeB, graph, &nodeA, 1, &paramsB);

  // Node 3: kernelC (add 100) depends on A (parallel with B)
  cudaGraphNode_t nodeC;
  float biasC = 100.0f;
  void* argsC[] = {&d_outC, &d_in, (void*)&n, &biasC};
  cudaKernelNodeParams paramsC = {};
  paramsC.func = (void*)add_k;
  paramsC.gridDim = dim3(blocks);
  paramsC.blockDim = dim3(threads);
  paramsC.kernelParams = argsC;
  cudaGraphAddKernelNode(&nodeC, graph, &nodeA, 1, &paramsC);

  // Instantiate and launch
  cudaGraphExec_t instance;
  cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaGraphLaunch(instance, stream);
  cudaStreamSynchronize(stream);

  // Verify: d_outB[i] = i * 2 + 10, d_outC[i] = i * 2 + 100
  float* h_outB = new float[n];
  float* h_outC = new float[n];
  cudaMemcpy(h_outB, d_outB, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_outC, d_outC, size, cudaMemcpyDeviceToHost);

  EXPECT_FLOAT_EQ(h_outB[0], 10.0f);   // 0 * 2 + 10
  EXPECT_FLOAT_EQ(h_outB[5], 20.0f);   // 5 * 2 + 10
  EXPECT_FLOAT_EQ(h_outC[0], 100.0f);  // 0 * 2 + 100
  EXPECT_FLOAT_EQ(h_outC[5], 110.0f);  // 5 * 2 + 100

  // Update kernel parameter in-place (change scale factor to 3)
  float newFactor = 3.0f;
  void* newArgsA[] = {&d_in, (void*)&n, &newFactor};
  paramsA.kernelParams = newArgsA;
  cudaGraphExecKernelNodeSetParams(instance, nodeA, &paramsA);

  for (int i = 0; i < n; i++) h_data[i] = static_cast<float>(i);
  cudaMemcpy(d_in, h_data, size, cudaMemcpyHostToDevice);
  cudaGraphLaunch(instance, stream);
  cudaStreamSynchronize(stream);

  cudaMemcpy(h_outB, d_outB, size, cudaMemcpyDeviceToHost);
  EXPECT_FLOAT_EQ(h_outB[5], 25.0f);  // 5 * 3 + 10

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
