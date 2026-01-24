====================
C/C++ Cheatsheet
====================

.. image:: https://img.shields.io/badge/doc-pdf-blue
   :target: https://cppcheatsheet.readthedocs.io/_/downloads/en/latest/pdf/

This cheatsheet provides a curated collection of C and C++ code snippets covering modern language standards, system programming, and development tools. From basic syntax to advanced features like coroutines, templates, and memory management, each example is designed to be clear, practical, and ready to use. All code is tested and compiles cleanly, so you can focus on learning and adapting rather than debugging reference material.

Modern C Programming
====================

Core C language features from C11 to C23, including memory management, preprocessor
macros, GNU extensions, and build systems.

- `C Basics <docs/notes/c/c_basic.rst>`_
- `Memory <docs/notes/c/c_memory.rst>`_
- `Preprocessor & GNU Extensions <docs/notes/c/c_macro.rst>`_
- `Makefile <docs/notes/c/make.rst>`_
- `X86 Assembly <docs/notes/c/asm.rst>`_

Modern C++ Programming
======================

Modern C++ features from C++11 to C++23. Covers resource management with RAII and
smart pointers, generic programming with templates and concepts, functional patterns
with lambdas, compile-time computation with constexpr, and asynchronous programming
with coroutines.

- `C++ Basics <docs/notes/cpp/cpp_basic.rst>`_
- `Resource Management <docs/notes/cpp/cpp_raii.rst>`_
- `String <docs/notes/cpp/cpp_string.rst>`_
- `Container <docs/notes/cpp/cpp_container.rst>`_
- `Iterator <docs/notes/cpp/cpp_iterator.rst>`_
- `Template <docs/notes/cpp/cpp_template.rst>`_
- `Casting <docs/notes/cpp/cpp_casting.rst>`_
- `Constexpr <docs/notes/cpp/cpp_constexpr.rst>`_
- `Lambda <docs/notes/cpp/cpp_lambda.rst>`_
- `Concepts <docs/notes/cpp/cpp_concepts.rst>`_
- `Requires <docs/notes/cpp/cpp_requires.rst>`_
- `Time <docs/notes/cpp/cpp_time.rst>`_
- `Smart Pointers <docs/notes/cpp/cpp_smartpointers.rst>`_
- `Return Value Optimization <docs/notes/cpp/cpp_rvo.rst>`_
- `Algorithm <docs/notes/cpp/cpp_algorithm.rst>`_
- `Coroutine <docs/notes/cpp/cpp_coroutine.rst>`_
- `Modules <docs/notes/cpp/cpp_modules.rst>`_
- `CMake <docs/notes/cpp/cpp_cmake.rst>`_

System Programming
==================

POSIX system programming covering process management, file I/O, signals, network
sockets, threading, and inter-process communication.

- `Process <docs/notes/os/os_process.rst>`_
- `File <docs/notes/os/os_file.rst>`_
- `Signal <docs/notes/os/os_signal.rst>`_
- `Socket <docs/notes/os/os_socket.rst>`_
- `Thread <docs/notes/os/os_thread.rst>`_
- `IPC <docs/notes/os/os_ipc.rst>`_

CUDA Programming
================

GPU programming with NVIDIA CUDA. Covers kernel basics, memory hierarchy, cooperative
groups, memory visibility, asynchronous pipelines, and multi-GPU communication.

- `CUDA Basics <docs/notes/cuda/cuda_basics.rst>`_
- `CUDA C++ (libcu++) <docs/notes/cuda/cuda_cpp.rst>`_
- `Thrust <docs/notes/cuda/cuda_thrust.rst>`_
- `Cooperative Groups <docs/notes/cuda/cuda_coop_groups.rst>`_
- `Memory Visibility <docs/notes/cuda/cuda_memory_visibility.rst>`_
- `Pipelines <docs/notes/cuda/cuda_pipelines.rst>`_
- `GPU-GPU Communication <docs/notes/cuda/cuda_ipc.rst>`_
- `Hardware Topology <docs/notes/cuda/cuda_hwloc.rst>`_

Bash & System Tools
===================

Command-line tools and shell scripting for system administration, networking,
and hardware inspection.

- `Bash <docs/notes/tools/bash.rst>`_
- `Operating System <docs/notes/tools/os.rst>`_
- `Network <docs/notes/tools/net.rst>`_
- `Hardware <docs/notes/tools/hardware.rst>`_
- `GPU <docs/notes/tools/gpu.rst>`_
- `Systemd <docs/notes/tools/systemd.rst>`_

Debugging & Profiling
=====================

Tools for debugging, memory analysis, and performance profiling of C/C++ and
CUDA applications.

- `GDB <docs/notes/debug/gdb.rst>`_
- `Valgrind <docs/notes/debug/valgrind.rst>`_
- `Sanitizers <docs/notes/debug/sanitizers.rst>`_
- `Tracing <docs/notes/debug/tracing.rst>`_
- `Perf <docs/notes/debug/perf.rst>`_
- `Nsight Systems <docs/notes/debug/nsight.rst>`_

Rust for C++ Developers
=======================

Rust programming guide for C++ developers. Maps C++ concepts to Rust equivalents
with side-by-side code comparisons covering ownership, traits, error handling,
smart pointers, and concurrency.

- `Basics <docs/notes/rust/rust_basic.rst>`_
- `Ownership & Borrowing <docs/notes/rust/rust_ownership.rst>`_
- `RAII & Drop <docs/notes/rust/rust_raii.rst>`_
- `Strings <docs/notes/rust/rust_string.rst>`_
- `Collections <docs/notes/rust/rust_container.rst>`_
- `Iterators <docs/notes/rust/rust_iterator.rst>`_
- `Traits & Generics <docs/notes/rust/rust_traits.rst>`_
- `Casting <docs/notes/rust/rust_casting.rst>`_
- `Const Functions <docs/notes/rust/rust_constfn.rst>`_
- `Closures <docs/notes/rust/rust_closure.rst>`_
- `Smart Pointers <docs/notes/rust/rust_smartptr.rst>`_
- `Error Handling <docs/notes/rust/rust_error.rst>`_
- `Threads <docs/notes/rust/rust_thread.rst>`_
- `Modules <docs/notes/rust/rust_modules.rst>`_
- `FFI & C++ Bindings <docs/notes/rust/rust_ffi.rst>`_
