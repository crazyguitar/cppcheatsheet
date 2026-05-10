// CUDA Graph - Conditional Nodes (CUDA 12.4+)
//
// This example demonstrates conditional nodes that enable if/else control
// flow within a graph. A device-side handle determines whether the body
// graph executes, removing the need to rebuild graphs for data-dependent
// branching.
//
// Key concepts:
// - cudaGraphConditionalHandleCreate() creates a device-side condition handle
// - cudaGraphCondTypeIf: body executes when handle value is non-zero
// - cudaGraphCondTypeWhile: body loops while handle value is non-zero
// - A kernel sets the handle value to control the branch at runtime
//
// Pitfalls:
// - Requires CUDA 12.4+ and compute capability 9.0+ (Hopper)
// - The condition handle must be set by a kernel BEFORE the conditional node
// - cudaGraphCondAssignDefault sets the initial value each launch
//
// Graph structure:
//   [set_condition]
//         |
//   [IF handle != 0]
//         |
//    ┌────┴────┐
//    │  body:  │
//    │[double] │
//    └─────────┘

#include <cuda_runtime.h>
#include <gtest/gtest.h>

__global__ void set_condition(cudaGraphConditionalHandle handle, int value) {
  if (threadIdx.x == 0) {
    cudaGraphSetConditional(handle, value);
  }
}

__global__ void double_val(float* data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] *= 2.0f;
}

TEST(CUDA, GraphConditionalNodes) {
  // Conditional nodes require compute capability 9.0+ (Hopper)
  int dev = 0;
  int major = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  if (major < 9) {
    GTEST_SKIP() << "Conditional nodes require compute capability 9.0+";
  }

  constexpr int n = 1024;
  constexpr size_t size = n * sizeof(float);
  constexpr int threads = 256;
  constexpr int blocks = (n + threads - 1) / threads;

  float *h_data, *d_data;
  cudaMallocHost(&h_data, size);
  cudaMalloc(&d_data, size);
  for (int i = 0; i < n; i++) h_data[i] = 1.0f;

  cudaGraph_t graph;
  cudaGraphCreate(&graph, 0);

  // Create conditional handle (default value = 0, body skipped)
  cudaGraphConditionalHandle handle;
  cudaGraphConditionalHandleCreate(&handle, graph, 0, cudaGraphCondAssignDefault);

  // Node 1: kernel sets condition based on runtime logic
  cudaGraphNode_t setNode;
  int condVal = 1;  // enable the branch
  void* setArgs[] = {&handle, &condVal};
  cudaKernelNodeParams setParams = {};
  setParams.func = (void*)set_condition;
  setParams.gridDim = dim3(1);
  setParams.blockDim = dim3(1);
  setParams.kernelParams = setArgs;
  cudaGraphAddKernelNode(&setNode, graph, nullptr, 0, &setParams);

  // Node 2: conditional IF node (depends on setNode)
  cudaGraphNodeParams condNodeParams = {};
  condNodeParams.type = cudaGraphNodeTypeConditional;
  condNodeParams.conditional.handle = handle;
  condNodeParams.conditional.type = cudaGraphCondTypeIf;
  condNodeParams.conditional.size = 1;

  cudaGraphNode_t condNode;
  cudaGraphAddNode(&condNode, graph, &setNode, 1, &condNodeParams);

  // Populate the body graph (executed when condition is non-zero)
  cudaGraph_t bodyGraph = condNodeParams.conditional.phGraph_out[0];
  cudaGraphNode_t bodyNode;
  void* bodyArgs[] = {&d_data, (void*)&n};
  cudaKernelNodeParams bodyParams = {};
  bodyParams.func = (void*)double_val;
  bodyParams.gridDim = dim3(blocks);
  bodyParams.blockDim = dim3(threads);
  bodyParams.kernelParams = bodyArgs;
  cudaGraphAddKernelNode(&bodyNode, bodyGraph, nullptr, 0, &bodyParams);

  // Instantiate and launch with condition=1 (body executes)
  cudaGraphExec_t instance;
  cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
  cudaGraphLaunch(instance, stream);
  cudaStreamSynchronize(stream);
  cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
  EXPECT_FLOAT_EQ(h_data[0], 2.0f);  // 1.0 * 2

  // Update condition to 0 (body skipped)
  int skipVal = 0;
  void* skipArgs[] = {&handle, &skipVal};
  setParams.kernelParams = skipArgs;
  cudaGraphExecKernelNodeSetParams(instance, setNode, &setParams);

  for (int i = 0; i < n; i++) h_data[i] = 1.0f;
  cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
  cudaGraphLaunch(instance, stream);
  cudaStreamSynchronize(stream);
  cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
  EXPECT_FLOAT_EQ(h_data[0], 1.0f);  // unchanged, body skipped

  cudaGraphExecDestroy(instance);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  cudaFreeHost(h_data);
  cudaFree(d_data);
}
