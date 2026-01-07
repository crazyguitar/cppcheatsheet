==============
Perf Reference
==============

.. meta::
   :description: Linux perf tutorial for CPU profiling, performance analysis, flame graphs, cache analysis, and system-wide tracing of C/C++ applications.
   :keywords: Linux perf, perf tutorial, CPU profiling, flame graph, performance analysis, perf record, perf report, cache misses, branch prediction, perf stat, system tracing, hotspot analysis

.. contents:: Table of Contents
    :backlinks: none

Introduction
------------

Linux perf is a powerful performance analysis tool that uses hardware performance
counters and kernel tracepoints to profile applications and the system. Unlike
sampling profilers that add overhead, perf leverages CPU hardware to count events
like cycles, instructions, cache misses, and branch mispredictions with minimal
impact on program execution.

Perf helps identify CPU hotspots, memory access patterns, and system bottlenecks.
It's essential for optimizing performance-critical code, understanding where time
is spent, and validating optimization efforts. The tool works on any executable
without recompilation, though debug symbols (``-g``) improve output readability.

Basic Profiling
---------------

Record and analyze CPU samples:

.. code-block:: bash

    $ perf record ./myprogram        # record CPU samples
    $ perf report                    # interactive report
    $ perf report --stdio            # text report

    $ perf record -g ./myprogram     # record with call graphs
    $ perf report -g                 # show call graph in report

Common record options:

.. code-block:: bash

    $ perf record -F 99 ./prog       # sample at 99 Hz
    $ perf record -p 1234            # attach to running process
    $ perf record -a sleep 10        # system-wide for 10 seconds
    $ perf record -o out.data ./prog # custom output file

Perf Stat
---------

Get summary statistics without recording samples:

.. code-block:: bash

    $ perf stat ./myprogram
     Performance counter stats for './myprogram':
          1,234.56 msec task-clock
               123 context-switches
         1,000,000 cycles
           800,000 instructions  #  0.80 insn per cycle
            50,000 cache-misses
            10,000 branch-misses

    $ perf stat -e cycles,instructions ./prog   # specific events
    $ perf stat -r 5 ./prog                     # run 5 times, show stats
    $ perf stat -d ./prog                       # detailed stats

Perf Top
--------

Real-time view of system or process hotspots:

.. code-block:: bash

    $ perf top                       # system-wide live view
    $ perf top -p 1234               # specific process
    $ perf top -F 99                 # sample at 99 Hz
    $ perf top -g                    # show call graphs
    $ perf top -ns comm,dso          # sort by process and library
    $ perf top -e cache-misses       # profile cache misses

Useful Events
-------------

Profile specific hardware and software events:

.. code-block:: bash

    # CPU events
    $ perf stat -e cycles,instructions,branches,branch-misses ./prog

    # Cache events
    $ perf stat -e cache-references,cache-misses ./prog
    $ perf stat -e L1-dcache-loads,L1-dcache-load-misses ./prog
    $ perf stat -e LLC-loads,LLC-load-misses ./prog

    # Memory events
    $ perf stat -e page-faults,minor-faults,major-faults ./prog

    # System calls
    $ perf stat -e 'syscalls:sys_enter_*' ./prog

List available events:

.. code-block:: bash

    $ perf list                      # all events
    $ perf list hw                   # hardware events
    $ perf list sw                   # software events
    $ perf list cache                # cache events
    $ perf list tracepoint           # kernel tracepoints

Call Graphs
-----------

Understand where time is spent in the call hierarchy:

.. code-block:: bash

    # Record with frame pointers (compile with -fno-omit-frame-pointer)
    $ perf record -g ./myprogram
    $ perf report -g

    # Record with DWARF unwinding (works without frame pointers)
    $ perf record --call-graph dwarf ./myprogram

    # Record with LBR (Last Branch Record, Intel CPUs)
    $ perf record --call-graph lbr ./myprogram

Flame Graphs
------------

Flame graphs visualize profiling data as interactive SVGs. The x-axis shows
stack depth, and width represents time spent. Download FlameGraph tools from
https://github.com/brendangregg/FlameGraph.

.. code-block:: bash

    $ perf record -g ./myprogram
    $ perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg

    # CPU flame graph
    $ perf record -F 99 -g ./prog
    $ perf script | stackcollapse-perf.pl | flamegraph.pl > cpu.svg

    # Off-CPU flame graph (shows where program waits)
    $ perf record -e sched:sched_switch -g ./prog

Annotate Source
---------------

See which source lines consume the most cycles:

.. code-block:: bash

    $ perf record ./myprogram
    $ perf annotate                  # interactive annotation
    $ perf annotate func_name        # annotate specific function
    $ perf annotate --stdio          # text output

Requires debug symbols (``-g``) and ideally compiled with ``-fno-omit-frame-pointer``.

System-Wide Analysis
--------------------

Profile the entire system to find bottlenecks:

.. code-block:: bash

    $ perf top -a                    # live system-wide view
    $ perf record -a -g sleep 30     # record system for 30 seconds
    $ perf report

    # Find which processes use most CPU
    $ perf top -ns comm

    # Trace context switches
    $ perf record -e context-switches -a sleep 10

Tracing
-------

Trace specific events and system calls:

.. code-block:: bash

    # Trace system calls
    $ perf trace ./myprogram
    $ perf trace -p 1234             # trace running process

    # Trace specific syscalls
    $ perf trace -e open,read,write ./prog

    # Count syscalls
    $ perf stat -e 'syscalls:sys_enter_*' ./prog

Comparing Runs
--------------

Compare performance between two runs:

.. code-block:: bash

    $ perf record -o before.data ./prog_v1
    $ perf record -o after.data ./prog_v2
    $ perf diff before.data after.data

Common Workflows
----------------

**Find CPU hotspots:**

.. code-block:: bash

    $ perf record -g ./myprogram
    $ perf report
    # Look for functions with highest "Overhead" percentage

**Diagnose cache performance:**

.. code-block:: bash

    $ perf stat -e cache-references,cache-misses,L1-dcache-load-misses ./prog
    # High cache-miss ratio indicates poor memory access patterns

**Profile a running server:**

.. code-block:: bash

    $ perf record -p $(pgrep myserver) -g sleep 30
    $ perf report

**Check if CPU-bound or I/O-bound:**

.. code-block:: bash

    $ perf stat ./myprogram
    # Low instructions-per-cycle + high context-switches = I/O bound
    # High instructions-per-cycle = CPU bound
