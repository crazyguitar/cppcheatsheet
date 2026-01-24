===========
CUDA Basics
===========

.. meta::
   :description: CUDA basics covering kernel launches, memory transfers, shared memory, streams, and common parallel patterns.
   :keywords: CUDA, GPU, kernel, shared memory, streams, reduction, matrix multiplication

.. contents:: Table of Contents
    :backlinks: none

CUDA enables massively parallel computation on NVIDIA GPUs. Programs consist of
host code (CPU) that launches kernel functions executing on the device (GPU).
Threads are organized into blocks, and blocks form a grid. Each thread identifies
itself using ``blockIdx`` and ``threadIdx`` to map to data elements.

Error Checking Macros
---------------------

CUDA API calls return error codes that should always be checked. Silent failures
lead to hard-to-debug issues where kernels produce wrong results or memory
operations fail without notice. These macros wrap API calls with automatic error
checking, printing the file, line number, and error description before exiting.
The ``LAUNCH_KERNEL`` macro uses ``cudaLaunchKernelEx`` which provides more
control over launch configuration than the ``<<<>>>`` syntax.

.. code-block:: cuda

    #define CUDA_CHECK(exp)                                                                                \
      do {                                                                                                 \
        cudaError_t err = (exp);                                                                           \
        if (err != cudaSuccess) {                                                                          \
          fprintf(stderr, "[%s:%d] %s failed: %s\n", __FILE__, __LINE__, #exp, cudaGetErrorString(err));   \
          exit(1);                                                                                         \
        }                                                                                                  \
      } while (0)

    #define LAUNCH_KERNEL(cfg, kernel, ...) CUDA_CHECK(cudaLaunchKernelEx(cfg, kernel, ##__VA_ARGS__))

    // Usage
    CUDA_CHECK(cudaMalloc(&d_data, size));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(blocks, 1, 1);
    cfg.blockDim = dim3(threads, 1, 1);
    cfg.stream = stream;
    LAUNCH_KERNEL(&cfg, my_kernel, d_data, n);

Device Properties
-----------------

:Source: `src/cuda/device-properties <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/device-properties>`_

Before launching kernels, query device capabilities using ``cudaGetDeviceProperties``.
This is essential for tuning kernel parameters like block size and shared memory usage
based on the hardware.

.. code-block:: cuda

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("Device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Total global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Registers per block: %d\n", prop.regsPerBlock);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads dim: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max grid size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Clock rate: %.2f MHz\n", prop.clockRate / 1000.0);
    printf("Memory clock rate: %.2f MHz\n", prop.memoryClockRate / 1000.0);
    printf("Memory bus width: %d bits\n", prop.memoryBusWidth);
    printf("L2 cache size: %d bytes\n", prop.l2CacheSize);
    printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
    printf("Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Unified addressing: %s\n", prop.unifiedAddressing ? "Yes" : "No");
    printf("Concurrent kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
    printf("ECC enabled: %s\n", prop.ECCEnabled ? "Yes" : "No");
    printf("PCI Domain:Bus:Device: %04x:%02x:%02x\n", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

Hello World
-----------

:Source: `src/cuda/hello <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/hello>`_

The ``__global__`` keyword declares a kernel function that runs on the GPU but
is callable from host code. The ``<<<blocks, threads>>>`` execution configuration
specifies how many thread blocks to launch and how many threads per block. Each
thread computes its unique global index using ``blockIdx.x * blockDim.x + threadIdx.x``,
which maps threads to data elements. Without ``cudaDeviceSynchronize()`` or stream
synchronization, the host continues execution immediately after launching the
kernel, potentially accessing results before they're ready.

.. code-block:: cuda

    __global__ void hello_kernel(int* output) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      output[idx] = idx;
    }

    // Launch: 2 blocks, 4 threads each = 8 total threads
    hello_kernel<<<2, 4>>>(d_output);
    cudaDeviceSynchronize();  // Wait for kernel to complete

Function Qualifiers
-------------------

CUDA provides function qualifiers to specify where functions execute and where
they can be called from. Understanding these is essential for organizing code
between host and device.

.. code-block:: cuda

    // __device__ - Device function: called from device, runs on device
    __device__ float square(float x) { return x * x; }

    // __host__ __device__ - Compiles for both host and device
    __host__ __device__ float add(float a, float b) { return a + b; }

    // __global__ - Kernel function: called from host, runs on device
    __global__ void compute(float* out, float a, float b) {
      out[0] = add(a, b);     // Call __host__ __device__ function
      out[1] = square(a);     // Call __device__ function
    }

Vector Addition
---------------

:Source: `src/cuda/vector-add <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/vector-add>`_

Vector addition demonstrates the most common CUDA pattern: one thread per data
element. Each thread independently computes one output value, achieving massive
parallelism with thousands of threads running simultaneously. The bounds check
``if (i < n)`` is critical because the total number of threads (blocks × threads
per block) is typically rounded up to a multiple of the block size, creating
extra threads that would access memory out of bounds. The ceiling division
formula ``(n + threads - 1) / threads`` ensures enough blocks are launched to
cover all elements.

.. code-block:: cuda

    __global__ void vector_add(const float* a, const float* b, float* c, int n) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) {  // Bounds check is critical
        c[i] = a[i] + b[i];
      }
    }

    int threads = 256;
    int blocks = (n + threads - 1) / threads;  // Ceiling division
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);

Memory Management
-----------------

:Source: `src/cuda/vector-add <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/vector-add>`_

CUDA provides several memory allocation methods with different performance
characteristics. Choosing the right one depends on access patterns and
transfer requirements.

**Device Memory** - Standard GPU memory. Fastest for GPU computation but
requires explicit transfers. Use for data that stays on GPU.

.. code-block:: cuda

    float* d_data;
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    cudaFree(d_data);

**Pinned (Page-Locked) Memory** - Host memory that cannot be paged out.
Required for ``cudaMemcpyAsync`` to truly overlap with computation. ~2x faster
transfers than pageable memory, but reduces available system memory.

.. code-block:: cuda

    float* h_pinned;
    cudaMallocHost(&h_pinned, size);
    cudaMemcpyAsync(d_data, h_pinned, size, cudaMemcpyHostToDevice, stream);
    cudaFreeHost(h_pinned);

**Host Registered Memory** - Pins existing malloc'd memory. Useful when you
can't change allocation code but need async transfers. Has registration
overhead, so avoid for short-lived buffers.

.. code-block:: cuda

    float* h_data = (float*)malloc(size);
    cudaHostRegister(h_data, size, cudaHostRegisterDefault);
    cudaHostUnregister(h_data);
    free(h_data);

**Unified Memory** - Single pointer accessible from both CPU and GPU. Driver
automatically migrates pages on demand. Simplifies code but may have
performance overhead from page faults. Best for prototyping or irregular
access patterns.

.. code-block:: cuda

    float* data;
    cudaMallocManaged(&data, size);
    kernel<<<blocks, threads>>>(data);   // GPU access
    cudaDeviceSynchronize();
    data[0] = 1.0f;                      // CPU access
    cudaFree(data);

**Zero-Copy Memory** - GPU directly reads/writes host memory over PCIe.
No explicit transfers needed, but each access has PCIe latency. Good for
data accessed once or when GPU memory is limited. Poor for repeated access.

.. code-block:: cuda

    float* h_mapped;
    float* d_mapped;
    cudaHostAlloc(&h_mapped, size, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_mapped, h_mapped, 0);
    kernel<<<blocks, threads>>>(d_mapped);
    cudaFreeHost(h_mapped);

**Summary** (64 MB benchmark, run ``cuda-memory-benchmark`` to test on your system):

+---------------------+------------------+------------------+------------------------+
| Method              | H2D Speed        | D2H Speed        | Use Case               |
+=====================+==================+==================+========================+
| Pageable (malloc)   | ~10.6 GB/s       | ~1.3 GB/s        | Simple, non-critical   |
+---------------------+------------------+------------------+------------------------+
| cudaMallocHost      | ~13.3 GB/s       | ~2.0 GB/s        | Async transfers        |
+---------------------+------------------+------------------+------------------------+
| cudaHostRegister    | ~13.2 GB/s       | ~2.1 GB/s        | Pin existing memory    |
+---------------------+------------------+------------------+------------------------+
| cudaMallocManaged   | ~0.8 GB/s        | ~0.4 GB/s        | Prototyping, irregular |
+---------------------+------------------+------------------+------------------------+

:Source: `src/cuda/memory-benchmark <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/memory-benchmark>`_

**RAII Pattern** - Ensures proper cleanup:

.. code-block:: cuda

    struct CudaBuffer {
      float* h_data;
      float* d_data;
      cudaStream_t stream;

      CudaBuffer(size_t size) {
        cudaStreamCreate(&stream);
        cudaMallocHost(&h_data, size);
        cudaMalloc(&d_data, size);
      }

      ~CudaBuffer() {
        cudaFree(d_data);
        cudaFreeHost(h_data);
        cudaStreamDestroy(stream);
      }
    };

Shared Memory
-------------

:Source: `src/cuda/matrix-mul <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/matrix-mul>`_
:Source: `src/cuda/shared-memory <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/shared-memory>`_

Shared memory is a programmer-managed cache located on the GPU chip, providing
much lower latency than global memory. All threads within a block share
the same shared memory space, making it ideal for data reuse patterns like tiled
matrix multiplication or stencil computations. The ``__shared__`` keyword declares
arrays in shared memory, and ``__syncthreads()`` creates a barrier ensuring all
threads complete their operations before proceeding. This synchronization is
essential because threads execute in warps of 32, and without barriers, some
threads may read data that others haven't written yet.

.. code-block:: cuda

    #define TILE_SIZE 16

    __global__ void tiled_kernel(float* data, int n) {
      __shared__ float tile[TILE_SIZE];  // Shared within block

      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      tile[threadIdx.x] = data[idx];  // Load to shared memory

      __syncthreads();  // Wait for all threads to finish loading

      // Now all threads can safely read from tile[]
      float result = tile[threadIdx.x] * 2.0f;
      data[idx] = result;
    }

Common pitfalls:

- Missing ``__syncthreads()`` causes race conditions
- Shared memory is limited (~48KB per block)
- Bank conflicts reduce performance

For complete examples including tiled matrix multiplication and stencil with
halo regions, see the source files linked above.

Parallel Reduction
------------------

:Source: `src/cuda/reduction <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/reduction>`_

Reduction is a fundamental parallel pattern for computing aggregates like sum,
min, max, or product across large arrays. The tree-based approach repeatedly
halves the number of active threads: in each iteration, thread ``i`` combines
its value with thread ``i + stride``. Shared memory holds intermediate results
within each block, and ``__syncthreads()`` ensures all threads complete each
level before proceeding to the next. After the kernel, each block has produced
one partial result, which can be combined on the CPU or with a second reduction
kernel. The block size should be a power of two for this simple implementation.

.. code-block:: cuda

    #define BLOCK_SIZE 256

    __global__ void reduce_sum(const float* input, float* output, int n) {
      __shared__ float sdata[BLOCK_SIZE];

      int tid = threadIdx.x;
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      sdata[tid] = (i < n) ? input[i] : 0.0f;
      __syncthreads();

      // Tree reduction: 128, 64, 32, 16, 8, 4, 2, 1
      for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
          sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
      }

      if (tid == 0) {
        output[blockIdx.x] = sdata[0];
      }
    }

Streams
-------

:Source: `src/cuda/streams <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/streams>`_

CUDA streams are independent execution queues that enable concurrent operations.
Operations within a stream execute in order, but operations in different streams
can overlap. This allows hiding memory transfer latency by pipelining: while one
chunk of data is being processed by a kernel, the next chunk can be transferred
to the GPU, and the previous chunk's results can be copied back to the host.
Pinned memory (``cudaMallocHost``) is mandatory for true asynchronous transfers;
pageable memory forces synchronous behavior even with ``cudaMemcpyAsync``. The
optimal number of streams depends on the workload and hardware, but 2-4 streams
typically provide good overlap without excessive scheduling overhead.

.. code-block:: cuda

    constexpr int num_streams = 4;
    cudaStream_t streams[num_streams];

    for (int i = 0; i < num_streams; i++) {
      cudaStreamCreate(&streams[i]);
    }

    // Pipeline: each stream handles a chunk independently
    for (int i = 0; i < num_streams; i++) {
      int offset = i * chunk_size;
      cudaMemcpyAsync(d_data + offset, h_data + offset, chunk_bytes, cudaMemcpyHostToDevice, streams[i]);
      kernel<<<blocks, threads, 0, streams[i]>>>(d_data + offset, chunk_size, i);
      cudaMemcpyAsync(h_data + offset, d_data + offset, chunk_bytes, cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < num_streams; i++) {
      cudaStreamSynchronize(streams[i]);
      cudaStreamDestroy(streams[i]);
    }
