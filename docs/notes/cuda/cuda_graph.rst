==========
CUDA Graph
==========

.. meta::
   :description: CUDA Graph API for capturing and replaying GPU work to reduce launch overhead and improve performance.
   :keywords: CUDA, GPU, graph, stream capture, kernel launch, optimization, replay

.. contents:: Table of Contents
    :backlinks: none

CUDA Graphs separate the definition of GPU work from its execution. Instead of
launching kernels one at a time, a graph captures a sequence of operations
(kernels, memory copies, etc.) and their dependencies, then replays the entire
graph with a single launch. This eliminates per-launch CPU overhead and enables
driver-level optimizations across the whole workflow.

Why CUDA Graphs
---------------

Traditional CUDA launches incur CPU overhead for each kernel dispatch. For
workloads with many small kernels, this overhead dominates execution time. Graphs
amortize this cost by submitting the entire workflow at once. The driver can also
optimize memory and scheduling across the graph.

.. code-block:: cuda

    // Without graphs: N kernel launches = N times CPU overhead
    for (int i = 0; i < N; i++) {
        kernel<<<grid, block, 0, stream>>>(args);
    }

    // With graphs: 1 launch for the entire workflow
    cudaGraphLaunch(instance, stream);

Stream Capture
--------------

:Source: `src/cuda/graph-stream-capture <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/graph-stream-capture>`_

Stream capture is the easiest way to create a graph. Existing stream-based code
can be wrapped with ``cudaStreamBeginCapture`` and ``cudaStreamEndCapture``
without modifying kernel launches. Operations issued to the captured stream are
recorded into a graph instead of being executed.

.. code-block:: cuda

    cudaGraph_t graph;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Begin capture
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // Record work (not executed yet)
    kernelA<<<grid, block, 0, stream>>>(d_data);
    kernelB<<<grid, block, 0, stream>>>(d_data);

    // End capture and get graph
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    // Instantiate and launch
    cudaGraphExec_t instance;
    CUDA_CHECK(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphLaunch(instance, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Cleanup
    cudaGraphExecDestroy(instance);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);

Capture mode ``cudaStreamCaptureModeGlobal`` prevents any non-captured stream
from launching work while capture is active, catching accidental work outside the
graph.

Explicit API
------------

:Source: `src/cuda/graph-explicit <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/graph-explicit>`_
:Source: `src/cuda/graph-explicit-1d <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/graph-explicit-1d>`_

The explicit API provides fine-grained control over graph topology. Nodes and
edges are added manually, which is useful when the dependency structure does not
map to a linear stream.

.. code-block:: cuda

    cudaGraph_t graph;
    CUDA_CHECK(cudaGraphCreate(&graph, 0));

    // Add kernel nodes
    cudaGraphNode_t nodeA, nodeB, nodeC;
    cudaKernelNodeParams paramsA = {};
    paramsA.func = (void *)kernelA;
    paramsA.gridDim = grid;
    paramsA.blockDim = block;
    paramsA.kernelParams = argsA;
    CUDA_CHECK(cudaGraphAddKernelNode(&nodeA, graph, nullptr, 0, &paramsA));

    cudaKernelNodeParams paramsB = {};
    paramsB.func = (void *)kernelB;
    paramsB.gridDim = grid;
    paramsB.blockDim = block;
    paramsB.kernelParams = argsB;
    // nodeB depends on nodeA
    CUDA_CHECK(cudaGraphAddKernelNode(&nodeB, graph, &nodeA, 1, &paramsB));

    cudaKernelNodeParams paramsC = {};
    paramsC.func = (void *)kernelC;
    paramsC.gridDim = grid;
    paramsC.blockDim = block;
    paramsC.kernelParams = argsC;
    // nodeC depends on nodeA (parallel with nodeB)
    CUDA_CHECK(cudaGraphAddKernelNode(&nodeC, graph, &nodeA, 1, &paramsC));

    // Instantiate and launch
    cudaGraphExec_t instance;
    CUDA_CHECK(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

This creates a diamond-like dependency: ``nodeA`` runs first, then ``nodeB`` and
``nodeC`` run in parallel.

Graph Update
------------

:Source: `src/cuda/graph-update <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/graph-update>`_

Updating an existing graph executable avoids the cost of destroying and
re-instantiating. Use ``cudaGraphExecUpdate`` to apply changes from a modified
graph to an existing executable. The update succeeds only if the graph topology
matches.

.. code-block:: cuda

    // Modify the original graph (e.g., change kernel parameters)
    cudaGraphExecUpdateResultInfo info;
    cudaGraphExecUpdateResult result;
    result = cudaGraphExecUpdate(instance, newGraph, &info);
    if (result != cudaGraphExecUpdateSuccess) {
        // Topology changed, must re-instantiate
        cudaGraphExecDestroy(instance);
        CUDA_CHECK(cudaGraphInstantiate(&instance, newGraph, nullptr, nullptr, 0));
    }

For simple parameter changes, ``cudaGraphExecKernelNodeSetParams`` updates a
single kernel node without rebuilding the graph.

.. code-block:: cuda

    // Update kernel parameters in-place
    paramsA.kernelParams = newArgs;
    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(instance, nodeA, &paramsA));

Memory Nodes
------------

:Source: `src/cuda/graph-memory-nodes <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/graph-memory-nodes>`_

CUDA 11.4+ supports ``cudaGraphAddMemAllocNode`` and ``cudaGraphAddMemFreeNode``
to allocate and free memory as part of the graph. This avoids pre-allocating
buffers and lets the driver optimize memory reuse across graph launches.

.. code-block:: cuda

    cudaMemAllocNodeParams allocParams = {};
    allocParams.poolProps.allocType = cudaMemAllocationTypePinned;
    allocParams.poolProps.location.type = cudaMemLocationTypeDevice;
    allocParams.poolProps.location.id = 0;
    allocParams.bytesize = N * sizeof(float);

    cudaGraphNode_t allocNode;
    CUDA_CHECK(cudaGraphAddMemAllocNode(&allocNode, graph, nullptr, 0, &allocParams));

    // The allocated pointer is available after instantiation
    float *d_buf = (float *)allocParams.dptr;

Conditional Nodes
-----------------

:Source: `src/cuda/graph-conditional <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/graph-conditional>`_

CUDA 12.4+ introduces conditional nodes that enable if/else and loop constructs
within a graph. A conditional node evaluates a device-side value to decide which
subgraph to execute, removing the need to re-capture or rebuild graphs for
data-dependent control flow.

.. code-block:: cuda

    cudaGraphConditionalHandle handle;
    CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, graph,
               1, // default value
               cudaGraphCondAssignDefault));

    // Create conditional node with IF type
    cudaGraphNodeParams cond_params = {};
    cond_params.type = cudaGraphNodeTypeConditional;
    cond_params.conditional.handle = handle;
    cond_params.conditional.type = cudaGraphCondTypeIf;
    cond_params.conditional.size = 1;

    cudaGraphNode_t condNode;
    CUDA_CHECK(cudaGraphAddNode(&condNode, graph, nullptr, 0, &cond_params));

    // Populate the body graph (executed when condition is non-zero)
    cudaGraph_t bodyGraph = cond_params.conditional.phGraph_out[0];

Best Practices
--------------

- Prefer stream capture over the explicit API for simpler code
- Reuse graph executables across iterations instead of re-instantiating
- Use ``cudaGraphExecUpdate`` for parameter changes with the same topology
- Profile with ``nsys`` to verify launch overhead reduction
- Avoid capturing operations with variable topology (use conditional nodes instead)
- Graph instantiation is expensive; do it once and replay many times
