=====================
GPU-GPU Communication
=====================

.. meta::
   :description: CUDA Inter-Process Communication (IPC) and P2P direct access for multi-GPU programming.
   :keywords: CUDA, IPC, P2P, NVLink, multi-GPU, cudaIpcGetMemHandle, cudaDeviceEnablePeerAccess

.. contents:: Table of Contents
    :backlinks: none

Introduction
------------

CUDA provides two mechanisms for GPU-to-GPU communication: IPC (Inter-Process
Communication) for multi-process scenarios and P2P Direct for single-process
multi-GPU applications. Both use the same hardware path (NVLink or PCIe) but
differ in their programming model.

- **P2P Direct**: Single process owns all GPUs. Use ``cudaDeviceEnablePeerAccess``
  to allow direct pointer dereferencing across GPUs. Simpler setup, no handle
  exchange needed. Best for single applications controlling all GPUs, simple
  multi-GPU parallelism, and low-latency scenarios.

- **IPC**: Multiple processes (e.g., MPI ranks) each own one GPU. Use
  ``cudaIpcGetMemHandle`` to export memory, exchange handles via MPI/sockets,
  then ``cudaIpcOpenMemHandle`` to access peer memory. Used by NCCL/NVSHMEM
  internally. Best for MPI-based distributed training and when fault isolation
  is needed.

Quick Reference
---------------

P2P Direct (Single-Process)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Single process owns all GPUs. Enable peer access to allow direct load/store to
peer GPU memory without explicit memcpy. Once enabled, kernels on GPU 0 can
dereference pointers allocated on GPU 1 as if they were local. The hardware
(NVLink or PCIe) handles the actual data transfer transparently.

.. code-block:: cuda

    // Single process with multiple GPUs
    //
    //   ┌─────────────────────────────────────────┐
    //   │ Process                                 │
    //   │ ┌─────────┐         ┌─────────┐         │
    //   │ │ GPU 0   │ ══════▶ │ GPU 1   │         │  ✓ Same address space
    //   │ │ ptr=A   │         │ ptr=B   │         │  ✓ Can dereference B from GPU 0
    //   │ └─────────┘         └─────────┘         │
    //   └─────────────────────────────────────────┘

    // Check and enable peer access
    int canAccess;
    cudaDeviceCanAccessPeer(&canAccess, gpu0, gpu1);
    if (canAccess) {
      cudaSetDevice(gpu0);
      cudaDeviceEnablePeerAccess(gpu1, 0);
    }

    // Now GPU 0 kernels can directly access GPU 1 memory
    // kernel<<<...>>>(gpu1_ptr);  // Direct access, no memcpy needed

    // Cleanup
    cudaDeviceDisablePeerAccess(gpu1);

IPC (Multi-Process)
~~~~~~~~~~~~~~~~~~~

Multiple processes (e.g., MPI ranks) each own one GPU. Since processes have
separate address spaces, raw pointers cannot be shared directly. Instead, export
a memory handle from one process, send it via MPI or shared memory, then import
it in another process to get a valid local pointer to the remote GPU memory.

.. code-block:: cuda

    // Multi-process (MPI) - each rank owns one GPU
    //
    //   ┌───────────────────┐   ┌───────────────────┐
    //   │ Process 0 (Rank 0)│   │ Process 1 (Rank 1)│
    //   │ ┌─────────┐       │   │  ┌─────────┐      │
    //   │ │ GPU 0   │ ─ ─ ─ │─ ─│─▶│ GPU 1   │      │  ✗ Different address spaces
    //   │ │ ptr=A   │       │   │  │ ptr=B   │      │  ✗ ptr B invalid in Process 0
    //   │ └─────────┘       │   │  └─────────┘      │
    //   └───────────────────┘   └───────────────────┘
    //   → Use IPC handles to share memory across processes

    // 1. Allocate and get handle
    int* d_buf;
    cudaMalloc(&d_buf, size);
    cudaIpcMemHandle_t handle;
    cudaIpcGetMemHandle(&handle, d_buf);

    // 2. Exchange handles via MPI
    MPI_Allgather(&handle, sizeof(handle), MPI_BYTE, all_handles, sizeof(handle), MPI_BYTE, MPI_COMM_WORLD);

    // 3. Open peer handles
    int* peer_buf;
    cudaIpcOpenMemHandle((void**)&peer_buf, peer_handle, cudaIpcMemLazyEnablePeerAccess);

    // 4. Use peer memory in kernels
    kernel<<<...>>>(peer_buf);

    // 5. Cleanup
    cudaIpcCloseMemHandle(peer_buf);

P2P Attributes
~~~~~~~~~~~~~~

Query P2P capabilities between GPUs to determine connection type and performance
characteristics. NVLink provides higher bandwidth and lower latency than PCIe,
and supports native atomics for efficient synchronization primitives.

.. code-block:: cuda

    int val;
    // Higher = better performance path
    cudaDeviceGetP2PAttribute(&val, cudaDevP2PAttrPerformanceRank, gpu0, gpu1);

    // Yes = NVLink, No = PCIe (slower)
    cudaDeviceGetP2PAttribute(&val, cudaDevP2PAttrNativeAtomicSupported, gpu0, gpu1);

    // Check if P2P access is supported at all
    cudaDeviceGetP2PAttribute(&val, cudaDevP2PAttrAccessSupported, gpu0, gpu1);

P2P Direct Example
------------------

:Source: `src/cuda/p2p-nvlink <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/p2p-nvlink>`_

Ring communication pattern where each GPU writes to the next GPU's buffer using
direct peer access. The example allocates buffers on each GPU, enables peer
access between all GPU pairs using ``cudaDeviceEnablePeerAccess``, then performs
a ring write where GPU i writes to GPU (i+1) % N. Verifies data integrity and
prints P2P attributes (NVLink vs PCIe) for each GPU pair.

IPC Example
-----------

:Source: `src/cuda/p2p-ipc <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/p2p-ipc>`_

Same ring communication pattern using IPC handles instead of direct peer access.
Uses threads to simulate multi-process behavior since the repo doesn't have MPI.
Each thread creates a buffer, exports an IPC handle via ``cudaIpcGetMemHandle``,
shares handles with other threads, opens peer handles via ``cudaIpcOpenMemHandle``,
writes to the next GPU's buffer, then verifies the data. Demonstrates the full
IPC lifecycle including proper cleanup with ``cudaIpcCloseMemHandle``.

Virtual Memory Management (VMM)
-------------------------------

:Source: `src/cuda/vmm <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/vmm>`_

VMM provides fine-grained control over GPU memory sharing using the CUDA Driver
API. Unlike ``cudaEnablePeerAccess`` which maps ALL ``cudaMalloc`` allocations
to peer devices, VMM lets you choose exactly which allocations to share. This is
the mechanism used internally by NCCL and NVSHMEM for selective memory sharing.
VMM separates physical memory allocation (``cuMemCreate``) from virtual address
mapping (``cuMemMap``), giving you explicit control over memory visibility.

.. code-block:: cuda

    // VMM: Fine-grained control over which allocations to share
    //
    //   cudaEnablePeerAccess:  Maps ALL cudaMalloc allocations to peer
    //   VMM (cuMemCreate):     Map only SPECIFIC allocations you choose
    //
    //   ┌─────────────────────────────────────────┐
    //   │ Process                                 │
    //   │ ┌─────────┐         ┌─────────┐         │
    //   │ │ GPU 0   │ ──────▶ │ GPU 1   │         │  Physical memory on GPU 0
    //   │ │ phys    │         │ mapped  │         │  Mapped to GPU 1's VA space
    //   │ └─────────┘         └─────────┘         │
    //   └─────────────────────────────────────────┘
    //
    //   cuMemCreate ──▶ cuMemAddressReserve ──▶ cuMemMap ──▶ cuMemSetAccess

    // 1. Allocate physical memory with shareable handle type
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    CUmemGenericAllocationHandle allocHandle;
    cuMemCreate(&allocHandle, size, &prop, 0);

    // 2. Export handle for sharing
    int fd;
    cuMemExportToShareableHandle(&fd, allocHandle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);

    // 3. Reserve virtual address range and map
    CUdeviceptr ptr;
    cuMemAddressReserve(&ptr, size, 0, 0, 0);
    cuMemMap(ptr, size, 0, allocHandle, 0);

    // 4. Set access permissions
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    cuMemSetAccess(ptr, size, &accessDesc, 1);

    // Now ptr is usable. Cleanup: cuMemUnmap, cuMemRelease, cuMemAddressFree

See `NVIDIA VMM Documentation <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html>`_
and `cuda-samples/memMapIPCDrv <https://github.com/NVIDIA/cuda-samples/tree/master/Samples/3_CUDA_Features/memMapIPCDrv>`_
for complete examples.
