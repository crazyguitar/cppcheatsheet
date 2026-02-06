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

.. note::
   All source code and benchmarks are available at `Libefaxx <https://github.com/crazyguitar/Libefaxx>`_.

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

Topology: GPU-NIC Affinity and NUMA-Aware Placement
----------------------------------------------------

Hardware topology awareness is essential for achieving optimal RDMA performance
in multi-GPU systems. On instances like AWS ``p5.48xlarge``, each GPU is
physically closer to certain EFA NICs and CPU cores through the PCIe topology.
Sending data through a nearby NIC avoids costly cross-NUMA or cross-PCIe-switch
transfers.

The diagram below illustrates this. If a process is bound to GPU 0, routing
RDMA traffic through the EFA device on the same PCIe switch minimizes latency.
Using a distant NIC (e.g., one closer to GPU 4) forces data to traverse
additional PCIe hops, increasing transfer time.

.. image:: ../../_static/blog/rdma/topology.png
   :alt: GPU-NIC PCIe topology diagram showing NUMA-aware placement of GPUs and EFA devices

Detecting Topology with hwloc
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One approach to discovering hardware topology is to parse
``/sys/bus/pci/devices`` directly, but this is error-prone and difficult to
maintain. A better approach is to use
`hwloc <https://github.com/open-mpi/hwloc>`_ — a portable library for
querying the hierarchical topology of CPUs, caches, NUMA nodes, and PCI
devices. The programming pattern resembles a DFS (pre-order traversal) over a
tree data structure. You can find basic usage examples in the
`hwloc cheat sheet <../cuda/cuda_hwloc.rst>`_ in this repository. For a real-world example of detecting GPU-NIC affinity on AWS ``p5.48xlarge``
and using ``taskset`` to pin processes to topology-local CPU cores, see
`affinity.h <https://github.com/crazyguitar/Libefaxx/blob/main/src/include/affinity/affinity.h>`_.

Bootstrap: Out-of-Band Connection Setup for RDMA
-------------------------------------------------

Unlike traditional networking stacks where protocols like ARP handle address
discovery automatically, RDMA requires an explicit out-of-band (OOB) exchange
to set up connections. Before any RDMA data transfer can occur, peers must
exchange endpoint addresses and memory region keys through a separate control
channel — a process known as **bootstrapping**.

Common bootstrap methods in the RDMA ecosystem include:

- **MPI** — NCCL and NVSHMEM can use MPI collectives (e.g.,
  ``MPI_Allgather``) to distribute connection identifiers such as
  ``nccl_id`` across all ranks.
- **TCP** — PyTorch's distributed runtime uses
  `TCPStore <https://pytorch.org/docs/stable/distributed.html#torch.distributed.TCPStore>`_
  as a key-value store to exchange connection information (e.g., rank
  addresses, NCCL IDs) between processes.

The diagram below illustrates the bootstrap flow:

.. image:: ../../_static/blog/rdma/bootstrap.png
   :alt: RDMA bootstrap sequence diagram showing out-of-band exchange of endpoint addresses and memory region keys

Once the RDMA connection is established and memory regions are registered, the
OOB channel is no longer needed for data transfer. In this post, the symmetric
memory implementation uses ``MPI_Allgather`` to exchange remote RDMA addresses
and memory region sizes — a straightforward approach compared to bootstrapping
via peer-to-peer RDMA calls. You can learn more details from `here <https://github.com/crazyguitar/Libefaxx/blob/main/src/include/bootstrap/mpi/fabric.h>`_.

Communication: RDMA Verbs and Data Transfer Patterns
-----------------------------------------------------

`libfabric` supports two primary communication patterns, each suited to different use
cases:

Two-Sided Communication (SEND/RECV)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This pattern resembles traditional TCP/IP socket communication. Both the sender
and receiver must actively participate — the sender calls ``fi_sendmsg`` and
the receiver calls ``fi_recvmsg``. Each side's completion queue (CQ) signals
when its respective operation completes. This is useful when the receiver needs
to know exactly when data arrives and control where it lands.

One-Sided Communication (RDMA WRITE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This pattern resembles a producer-consumer model with shared memory. The writer
uses ``fi_writemsg`` to write directly into the remote node's registered memory
region (MR) — the remote CPU is not involved in the data path. Only the
writer's CQ signals completion; the remote side has no automatic notification
that data arrived.

To notify the remote side, RDMA provides **write with immediate data**
(``FI_REMOTE_CQ_DATA``). The writer attaches a small immediate value to the
write operation. When the write completes, the remote CQ receives a completion
event containing this immediate data, signaling that new data is available.
This is commonly used as an "end-of-write" tag.

.. image:: ../../_static/blog/rdma/comm.png
   :alt: RDMA communication patterns diagram comparing two-sided SEND/RECV with one-sided WRITE with immediate data

Implementation examples for both patterns are available in Libefaxx:
`Send/Recv benchmark <https://github.com/crazyguitar/Libefaxx/tree/main/experiments/sendrecv>`_ and
`Write benchmark <https://github.com/crazyguitar/Libefaxx/tree/main/experiments/write>`_.

Why NVSHMEM Uses One-Sided Semantics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At first glance, the name "NVSHMEM" (and OpenSHMEM) might suggest it is just
another shared memory IPC library. However, NVSHMEM also supports inter-node
communication over RDMA. The "SHMEM" terminology reflects the programming
model: like shared memory IPC, the communication is one-sided — a producer
writes to a remote address without the consumer explicitly receiving. RDMA's
one-sided write maps naturally to this model: you specify a remote virtual
address and offset, and the NIC performs a DMA-like transfer directly into
remote memory. This is why one-sided RDMA is the foundation for NVSHMEM's
``nvshmem_put`` / ``nvshmem_get`` APIs.

Reference
---------

1. Le, Q., "Libfabric EFA Series," 2024. `[link] <https://le.qun.ch/en/blog/2024/12/25/libfabric-efa-0-intro/>`_
2. Punniyamurthy, K. et al., "Optimizing Distributed ML Communication," arXiv:2305.06942, 2023. `[arXiv] <https://arxiv.org/pdf/2305.06942>`_
3. Liu, S. et al., "GPU-Initiated Networking," arXiv:2511.15076, 2025. `[arXiv] <https://arxiv.org/abs/2511.15076>`_
4. UCCL Project, "UCCL: User-space Collective Communication Library." `[GitHub] <https://github.com/uccl-project/uccl>`_
5. Microsoft, "MSCCL++: Multi-Scale Collective Communication Library." `[GitHub] <https://github.com/microsoft/mscclpp>`_
6. DeepSeek-AI, "DeepEP: Expert parallelism with GPU-initiated communication." `[GitHub] <https://github.com/deepseek-ai/DeepEP>`_
