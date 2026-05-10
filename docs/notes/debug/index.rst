=====================
Debugging & Profiling
=====================

.. meta::
   :description: C/C++ debugging and profiling guide covering GDB, Valgrind, sanitizers (ASan/TSan/UBSan), Linux tracing (strace/ltrace/ftrace), perf profiling, and NVIDIA Nsight GPU analysis.
   :keywords: GDB debugger, Valgrind memcheck, AddressSanitizer, ThreadSanitizer, strace, ltrace, Linux perf, Nsight Systems, Nsight Compute, CUDA debugging, memory leak detection, CPU profiling

When software misbehaves or underperforms, the ability to inspect program state and measure
execution characteristics becomes invaluable. Debugging and profiling transform opaque failures
into understandable problems with actionable solutions.

This section covers essential tools for C/C++ development: GDB for interactive debugging,
Valgrind for memory error detection, compiler sanitizers for fast runtime checks, tracing
tools for system call analysis, perf for CPU profiling, and NVIDIA Nsight for GPU optimization.

.. toctree::
   :maxdepth: 1

   gdb
   valgrind
   sanitizers
   tracing
   perf
   nsight
   binary_tools
