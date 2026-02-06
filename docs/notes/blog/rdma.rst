=========================================================
Building NVSHMEM from Scratch: GPU-Initiated Networking
=========================================================

.. meta::
   :description: Learn how to implement NVSHMEM-like GPU-initiated networking using proxy threads for RDMA, GPUDirect, and distributed LLM training.
   :keywords: NVSHMEM, RDMA, GPUDirect, InfiniBand, NCCL, GPU communication, LLM training, distributed deep learning, MoE, DeepEP, CUDA

Abstract
--------

GPU-to-GPU communication is critical for LLM training and inference, as many
large language models cannot fit on a single GPU or even a single node. This
requires partitioning model parameters across GPUs and using collective
communication to aggregate results. NCCL is a widely-used collective library
that achieves high efficiency over RDMA fabrics like InfiniBand or AWS Elastic
Fabric Adapter (EFA) for exchanging data between GPUs. Recently, GPU-Initiated
Networking (GIN) has gained attention for its ability to fuse CUDA kernels with
GPUDirect communication. DeepEP exemplifies this approach—a high-performance
MoE layer dispatch/combine implementation that significantly reduces All-to-All
collective latency using NVSHMEM with InfiniBand GPUDirect Async (IBGDA).
However, not all RDMA providers support IBGDA (at least as of late 2025).
Instead, they rely on a "Proxy Thread" technique to achieve GIN. InfiniBand
Reliable Connection (IBRC) uses this approach, as do similar implementations
like UCCL and MSCCL++. In this post, we break down how proxy thread solutions
achieve GPU-initiated behavior over AWS EFA similar to NVSHMEM.

Key Components for Building NVSHMEM
------------------------------------

Before diving into the implementation, let's outline the essential libraries,
data structures, and algorithms required. The following components form the
foundation for building a minimal NVSHMEM-like system:

1. **EFA Library (libfabric)** – RDMA transport abstraction layer
2. **Hardware Topology (hwloc)** – GPU-NIC affinity and NUMA-aware placement
3. **RDMA Bootstrap** – Connection setup and endpoint exchange
4. **GPUDirect RDMA (DMA-BUF)** – Zero-copy GPU memory registration for RDMA
5. **Symmetric Memory** – Globally addressable memory across GPUs
6. **GPU-CPU Queue (GDRCopy / Unified Memory / Pinned Memory)** – Low-latency signaling between GPU kernels and CPU proxy
7. **Proxy Thread** – CPU-side thread that issues RDMA operations on behalf of GPU kernels
8. **Communication Patterns (SEND/RECV/WRITE)** – RDMA verbs for data transfer
9. **CUDA IPC** – Intra-node GPU-to-GPU communication via shared memory
10. **NVSHMEM Implementation** – Putting it all together into a GPU-initiated networking layer

.. image:: /_static/blog/rdma/fabric.png

.. image:: /_static/blog/rdma/topology.png

.. image:: /_static/blog/rdma/bootstrap.png

.. image:: /_static/blog/rdma/comm.png
