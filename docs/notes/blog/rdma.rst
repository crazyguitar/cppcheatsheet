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
achieve GPU-initiated behavior over AWS EFA similar to NVSHMEM. You can find
the all experiments and benchmark results in `Libefaxx <https://github.com/crazyguitar/Libefaxx>`_.

Key Components for Building NVSHMEM
------------------------------------

Before diving into the implementation, let's outline the essential libraries,
data structures, and algorithms required. The following components form the
foundation for building a minimal NVSHMEM-like library:

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

Fabric: RDMA Transport with libfabric and AWS EFA
--------------------------------------------------

To perform RDMA operations, applications typically use low-level libraries such
as `libibverbs <https://github.com/linux-rdma/rdma-core/tree/master>`_ (for
InfiniBand/RoCE) or `libfabric <https://github.com/ofiwg/libfabric>`_ (a
higher-level, provider-agnostic fabric interface). Since this post targets AWS
Elastic Fabric Adapter (EFA), we use ``libfabric`` — the recommended interface
for EFA. The AWS EFA provider in libfabric handles the
`Scalable Reliable Datagram (SRD) <https://aws.amazon.com/blogs/hpc/in-the-search-for-performance-theres-more-than-one-way-to-build-a-network/>`_
protocol internally, so applications do not need to manage reliability or
ordering at the transport layer.

libfabric Object Hierarchy
^^^^^^^^^^^^^^^^^^^^^^^^^^

The diagram below illustrates the core libfabric object hierarchy used to set
up RDMA communication over EFA:

- **Fabric** — represents the physical network (e.g., an EFA device).
- **Domain** — maps to a specific network interface, similar to binding to an
  IP address. Each domain provides access to resources such as memory
  registration and address resolution.
- **Endpoint** — a communication channel, analogous to a socket or port. Each
  endpoint is associated with:

  - An **Address Vector (AV)** — a table that maps peer addresses for
    connectionless (datagram) communication.
  - A **Completion Queue (CQ)** — used to poll for RDMA operation completions
    (e.g., send/recv/write done).

The full initialization sequence can be found in
`efa.h <https://github.com/crazyguitar/Libefaxx/blob/main/src/include/rdma/fabric/efa.h#L199-L213>`_.

.. image:: ../../_static/blog/rdma/fabric.png
   :alt: libfabric object hierarchy diagram showing fabric, domain, endpoint, address vector, and completion queue relationships

Querying EFA Devices with fi_info
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On AWS EC2 instances with EFA enabled (e.g., ``p4d.24xlarge``,
``p5.48xlarge``), you can query available fabric providers using the
``fi_info`` utility:

.. code-block:: bash

    $ fi_info -p efa

    provider: efa
        fabric: efa
        domain: rdmap201s0-dgrm
        version: 203.10
        type: FI_EP_DGRAM
        protocol: FI_PROTO_EFA

The output shows that the EFA provider exposes a datagram endpoint
(``FI_EP_DGRAM``) using the ``FI_PROTO_EFA`` protocol. The ``domain`` field
identifies the specific EFA device. On multi-NIC instances (e.g.,
``p5.48xlarge`` with 32 EFA interfaces), ``fi_info`` will list multiple
domains — one per NIC — which is important for topology-aware placement
discussed in the next section.

Topology
--------

.. image:: ../../_static/blog/rdma/topology.png

Bootstrap
---------

.. image:: ../../_static/blog/rdma/bootstrap.png

Communication
-------------

.. image:: ../../_static/blog/rdma/comm.png

Reference
---------
