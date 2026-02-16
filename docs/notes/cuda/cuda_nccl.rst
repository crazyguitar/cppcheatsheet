====
NCCL
====

.. meta::
   :description: NCCL (NVIDIA Collective Communications Library) for multi-GPU collective operations including AllReduce, AllGather, Broadcast, Reduce, and ReduceScatter.
   :keywords: NCCL, CUDA, multi-GPU, collective, AllReduce, AllGather, Broadcast, Reduce, ReduceScatter, distributed, MPI

.. contents:: Table of Contents
    :backlinks: none

NCCL (pronounced "Nickel") is NVIDIA's library for multi-GPU and multi-node
collective communication. It implements optimized primitives like AllReduce,
Broadcast, and AllGather that automatically exploit the topology — NVLink, PCIe,
InfiniBand — to maximize bandwidth. NCCL is the communication backbone of most
distributed deep learning frameworks (PyTorch DDP, Megatron-LM, DeepSpeed).

Bootstrap
---------

Before any collective can run, every participating GPU needs a communicator.
NCCL provides two initialization paths depending on whether you run single-process
or multi-process.

Single-Process (ncclCommInitAll)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When one process owns all GPUs on a single node, ``ncclCommInitAll`` creates
communicators for every device in one call. No ID exchange is needed.

.. code-block:: cuda

    int nDev;
    cudaGetDeviceCount(&nDev);
    std::vector<ncclComm_t> comms(nDev);
    std::vector<int> devs(nDev);
    std::iota(devs.begin(), devs.end(), 0);  // {0, 1, ..., nDev-1}
    ncclCommInitAll(comms.data(), nDev, devs.data());

This is the simplest setup and what all the examples below use.

Multi-Process with MPI (ncclCommInitRank)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For multi-node or one-GPU-per-process setups, rank 0 generates a unique ID and
broadcasts it to all ranks via MPI (or any out-of-band channel). Each rank then
calls ``ncclCommInitRank`` with the shared ID.

.. code-block:: cuda

    #include <mpi.h>
    #include <nccl.h>

    int rank, nRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    // Rank 0 generates the unique ID
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);

    // Broadcast ID to all ranks (ncclUniqueId is a 128-byte opaque struct)
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Each rank selects its GPU and initializes its communicator
    cudaSetDevice(rank % numLocalGPUs);
    ncclComm_t comm;
    ncclCommInitRank(&comm, nRanks, id, rank);

The ``ncclUniqueId`` can be exchanged via any mechanism — MPI, TCP sockets,
shared filesystem, etc. MPI is the most common choice because distributed GPU
workloads typically already use MPI for process management.

AllReduce
---------

:Source: `src/cuda/nccl-allreduce <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/nccl-allreduce>`_

AllReduce combines values from all ranks with a reduction operation and
distributes the result back to every rank. This is the most common collective in
distributed training — used to synchronize gradients across data-parallel
workers.

.. code-block:: cuda

    // Before:  rank0=[1,2,3]  rank1=[4,5,6]
    // After:   rank0=[5,7,9]  rank1=[5,7,9]  (element-wise sum)
    //
    // Supports ncclSum, ncclProd, ncclMax, ncclMin, ncclAvg
    // sendbuff and recvbuff can be the same pointer (in-place)

    constexpr int count = 1024;
    std::vector<float> h_send(count);
    std::iota(h_send.begin(), h_send.end(), 0.0f);  // [0, 1, 2, ...]
    float *d_send, *d_recv;
    cudaSetDevice(devId);
    cudaMalloc(&d_send, count * sizeof(float));
    cudaMalloc(&d_recv, count * sizeof(float));
    cudaMemcpy(d_send, h_send.data(), count * sizeof(float), cudaMemcpyHostToDevice);

    ncclAllReduce(d_send, d_recv, count, ncclFloat, ncclSum, comm, stream);
    cudaStreamSynchronize(stream);

Broadcast
---------

:Source: `src/cuda/nccl-broadcast <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/nccl-broadcast>`_

Broadcast copies data from one root rank to all other ranks. Useful for
distributing model weights or configuration data at initialization.

.. code-block:: cuda

    // Before:  rank0=[1,2,3]  rank1=[0,0,0]
    // After:   rank0=[1,2,3]  rank1=[1,2,3]  (root=0)
    //
    // On root, sendbuff is the source. On non-root, sendbuff is ignored.

    constexpr int count = 1024;
    constexpr int root = 0;
    std::vector<float> h_buf(count);
    std::iota(h_buf.begin(), h_buf.end(), 0.0f);  // [0, 1, 2, ...]
    float *d_buf;
    cudaSetDevice(devId);
    cudaMalloc(&d_buf, count * sizeof(float));
    if (devId == root) cudaMemcpy(d_buf, h_buf.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    ncclBroadcast(d_buf, d_buf, count, ncclFloat, root, comm, stream);
    cudaStreamSynchronize(stream);

Reduce
------

:Source: `src/cuda/nccl-reduce <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/nccl-reduce>`_

Reduce combines values from all ranks but stores the result only on the root
rank. Other ranks' receive buffers are untouched.

.. code-block:: cuda

    // Before:  rank0=[1,2,3]  rank1=[4,5,6]
    // After:   rank0=[5,7,9]  rank1=[...]    (root=0, only root gets result)

    constexpr int count = 1024;
    constexpr int root = 0;
    std::vector<float> h_send(count, static_cast<float>(rank + 1));
    float *d_send, *d_recv;
    cudaSetDevice(devId);
    cudaMalloc(&d_send, count * sizeof(float));
    cudaMalloc(&d_recv, count * sizeof(float));
    cudaMemcpy(d_send, h_send.data(), count * sizeof(float), cudaMemcpyHostToDevice);

    ncclReduce(d_send, d_recv, count, ncclFloat, ncclSum, root, comm, stream);
    cudaStreamSynchronize(stream);

AllGather
---------

:Source: `src/cuda/nccl-allgather <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/nccl-allgather>`_

AllGather concatenates each rank's buffer into a single output on every rank.
Each rank contributes ``sendcount`` elements and receives ``sendcount * nRanks``
elements.

.. code-block:: cuda

    // Before:  rank0=[A]      rank1=[B]
    // After:   rank0=[A,B]    rank1=[A,B]

    constexpr int sendcount = 512;
    int recvcount = sendcount * nRanks;
    std::vector<float> h_send(sendcount, static_cast<float>(rank));
    float *d_send, *d_recv;
    cudaSetDevice(devId);
    cudaMalloc(&d_send, sendcount * sizeof(float));
    cudaMalloc(&d_recv, recvcount * sizeof(float));  // nRanks * sendcount
    cudaMemcpy(d_send, h_send.data(), sendcount * sizeof(float), cudaMemcpyHostToDevice);

    ncclAllGather(d_send, d_recv, sendcount, ncclFloat, comm, stream);
    cudaStreamSynchronize(stream);

ReduceScatter
-------------

:Source: `src/cuda/nccl-reducescatter <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/nccl-reducescatter>`_

ReduceScatter reduces across ranks then scatters equal chunks to each rank. It
is the inverse of AllGather — each rank ends up with a reduced slice. This is
used in ZeRO-style optimizers where each rank owns a shard of the parameters.

.. code-block:: cuda

    // Before:  rank0=[1,2,3,4]  rank1=[5,6,7,8]  (2 ranks, 4 elements)
    // After:   rank0=[6,8]      rank1=[10,12]     (sum, 2 elements each)

    constexpr int totalCount = 1024;
    int recvcount = totalCount / nRanks;
    std::vector<float> h_send(totalCount);
    std::iota(h_send.begin(), h_send.end(), 0.0f);  // [0, 1, 2, ...]
    float *d_send, *d_recv;
    cudaSetDevice(devId);
    cudaMalloc(&d_send, totalCount * sizeof(float));
    cudaMalloc(&d_recv, recvcount * sizeof(float));  // totalCount / nRanks
    cudaMemcpy(d_send, h_send.data(), totalCount * sizeof(float), cudaMemcpyHostToDevice);

    ncclReduceScatter(d_send, d_recv, recvcount, ncclFloat, ncclSum, comm, stream);
    cudaStreamSynchronize(stream);

Grouping Collectives
--------------------

:Source: `src/cuda/nccl-alltoall <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/nccl-alltoall>`_

When multiple operations need to run as a batch, wrapping them in a group lets
NCCL fuse them into fewer kernel launches. This is required when issuing
point-to-point calls from a single thread, and recommended for performance in
general.

A common use case is building AlltoAll out of grouped Send/Recv. NCCL does not
provide a dedicated AlltoAll primitive, but it can be composed:

.. code-block:: cuda

    // Naive AlltoAll: each rank sends a chunk to every other rank
    //
    // Before:  rank0=[A0,A1]  rank1=[B0,B1]   (2 ranks, chunk per peer)
    // After:   rank0=[A0,B0]  rank1=[A1,B1]

    constexpr int chunkCount = 512;
    size_t chunkSize = chunkCount * sizeof(float);

    // d_send: nRanks chunks laid out contiguously [chunk_for_rank0, chunk_for_rank1, ...]
    // d_recv: nRanks chunks to receive into
    float *d_send, *d_recv;
    cudaMalloc(&d_send, nRanks * chunkSize);
    cudaMalloc(&d_recv, nRanks * chunkSize);
    cudaMemcpy(d_send, h_send, nRanks * chunkSize, cudaMemcpyHostToDevice);

    ncclGroupStart();
    for (int peer = 0; peer < nRanks; peer++) {
        ncclSend(d_send + peer * chunkCount, chunkCount, ncclFloat, peer, comm, stream);
        ncclRecv(d_recv + peer * chunkCount, chunkCount, ncclFloat, peer, comm, stream);
    }
    ncclGroupEnd();
    cudaStreamSynchronize(stream);

All calls between ``ncclGroupStart`` and ``ncclGroupEnd`` are batched into a
single operation. Without grouping, each Send/Recv pair would deadlock waiting
for its matching peer.

Best Practices
--------------

- Always wrap multi-communicator calls in ``ncclGroupStart``/``ncclGroupEnd``
- Use in-place operations (``sendbuff == recvbuff``) to save memory
- Pin host memory with ``cudaMallocHost`` for staging buffers
- Match NCCL calls 1:1 across all ranks — mismatched calls deadlock
- Use ``NCCL_DEBUG=INFO`` environment variable to diagnose topology and ring selection
- Prefer ``ncclAllReduce`` with ``ncclAvg`` over manual sum + divide for gradient averaging
- Capture NCCL collectives into CUDA graphs for repeated operations to eliminate launch overhead

CUDA Graph Capture
------------------

NCCL collectives can be captured into CUDA graphs via stream capture, eliminating
per-iteration kernel launch overhead. This is particularly effective for training
loops where the same collective pattern repeats thousands of times. NCCL >= 2.15.1
and CUDA >= 11.7 are required.

Stream capture only **records** the operations — no kernels are executed during
capture. The work is deferred until ``cudaGraphLaunch`` replays the graph.

The pattern is: warmup the collective first (NCCL needs to establish internal
state), then capture into a graph, instantiate, and replay. A small RAII wrapper
makes this clean:

.. code-block:: cuda

    struct CUDAGraph {
      cudaGraphExec_t exec_ = nullptr;

      template <typename F>
      CUDAGraph(cudaStream_t s, F&& fn) {
        cudaGraph_t g;
        cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
        fn();
        cudaStreamEndCapture(s, &g);
        cudaGraphInstantiate(&exec_, g, nullptr, nullptr, 0);
        cudaGraphDestroy(g);
      }

      ~CUDAGraph() { if (exec_) cudaGraphExecDestroy(exec_); }
      void launch(cudaStream_t s) { cudaGraphLaunch(exec_, s); }
    };

Usage with a single collective:

.. code-block:: cuda

    // Warmup — NCCL initializes internal buffers on first call
    ncclAllReduce(d_send, d_recv, count, ncclFloat, ncclSum, comm, stream);
    cudaStreamSynchronize(stream);

    // Capture and replay
    CUDAGraph graph(stream, [&] {
        ncclAllReduce(d_send, d_recv, count, ncclFloat, ncclSum, comm, stream);
    });

    for (int iter = 0; iter < num_iters; iter++) graph.launch(stream);
    cudaStreamSynchronize(stream);

Grouped operations (``ncclGroupStart``/``ncclGroupEnd``) are also capturable.
The entire group becomes a single graph node:

.. code-block:: cuda

    CUDAGraph graph(stream, [&] {
        ncclGroupStart();
        for (int peer = 0; peer < nRanks; peer++) {
            ncclSend(d_send + peer * chunk, chunk, ncclFloat, peer, comm, stream);
            ncclRecv(d_recv + peer * chunk, chunk, ncclFloat, peer, comm, stream);
        }
        ncclGroupEnd();
    });

For multi-GPU single-process setups, each device needs its own graph captured on
its own stream. All graphs are then launched and synchronized independently.

.. note::

    When using ``ncclMemAlloc`` for user buffer registration (NCCL >= 2.19.1),
    ``ncclCommRegister`` is not required if NCCL kernels are launched from a
    CUDA graph captured via stream capture — NCCL handles registration
    automatically during capture.
