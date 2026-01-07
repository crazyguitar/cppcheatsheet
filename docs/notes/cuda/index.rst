================
CUDA Programming
================

.. meta::
   :description: CUDA programming guide covering kernel launches, memory management, shared memory optimization, streams, and parallel reduction patterns.
   :keywords: CUDA, GPU programming, NVIDIA, parallel computing, shared memory, streams, reduction, matrix multiplication

CUDA is NVIDIA's parallel computing platform that enables developers to harness
the massive parallelism of GPUs for general-purpose computing. Unlike CPUs
optimized for sequential tasks with a few powerful cores, GPUs contain thousands
of smaller cores designed for throughput-oriented workloads. This architecture
excels at data-parallel problems where the same operation is applied to many
elements simultaneously, such as matrix operations, image processing, and
scientific simulations.

The CUDA programming model extends C++ with keywords for defining kernel
functions that execute on the GPU and APIs for managing memory transfers between
host (CPU) and device (GPU). Understanding memory hierarchy is crucial for
performance: global memory has high bandwidth but high latency, while shared
memory provides fast on-chip storage shared within thread blocks. Effective CUDA
programming involves minimizing data transfers, maximizing parallelism, and
optimizing memory access patterns.

.. toctree::
   :maxdepth: 1

   cuda_basics
