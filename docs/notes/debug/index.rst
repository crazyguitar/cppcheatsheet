Debugging & Profiling
=====================

.. meta::
   :description: C/C++ debugging and performance profiling guide covering GDB debugger commands, breakpoints, memory inspection, and Linux perf analysis tools.
   :keywords: GDB debugger, debugging C++, performance profiling, Linux perf, breakpoints, memory debugging, stack traces, code optimization

When software misbehaves or underperforms, the ability to inspect program state and measure
execution characteristics becomes invaluable. Debugging and profiling transform opaque failures
into understandable problems with actionable solutions.

GDB, the GNU Debugger, enables interactive examination of running programs—setting breakpoints,
inspecting variables, navigating call stacks, and detecting memory corruption. The Linux perf
tool complements debugging with performance analysis, revealing CPU hotspots, cache misses, and
call graph bottlenecks. Together, these tools are essential for delivering correct, efficient
software.

.. toctree::
   :maxdepth: 1

   gdb_debug
   perf
