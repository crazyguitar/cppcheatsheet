========================
Nsight Systems Reference
========================

.. meta::
   :description: NVIDIA Nsight Systems tutorial for GPU profiling, CPU-GPU timeline visualization, CUDA kernel analysis, memory transfer optimization, and application performance analysis.
   :keywords: Nsight Systems, CUDA profiling, GPU profiling, NVIDIA profiler, GPU timeline, CUDA optimization, GPU performance, NVTX, nsys, CPU-GPU overlap, memory transfer

.. contents:: Table of Contents
    :backlinks: none

Introduction
------------

NVIDIA Nsight Systems provides system-wide performance analysis for GPU
applications, showing CPU activity, GPU kernels, memory transfers, and API
calls on a unified timeline. Understanding GPU performance requires visibility
into both the CPU-side code that launches kernels and the GPU-side execution
that processes data—Nsight Systems captures both simultaneously.

The timeline view reveals how CPU and GPU activities overlap and interact,
exposing idle time, serialization bottlenecks, and opportunities for better
concurrency. You can see exactly when kernels launch, how long memory transfers
take, and where the CPU waits for GPU completion. This holistic view is essential
for optimizing the overall application, not just individual kernels.

Nsight Systems works with any CUDA application without requiring source code
modifications, though adding NVTX annotations improves the clarity of timeline
visualizations. It supports both command-line profiling for automated workflows
and GUI visualization for interactive analysis.

Basic Profiling
---------------

Command-line profiling captures a trace that can be analyzed later:

.. code-block:: bash

    $ nsys profile ./myprogram
    $ nsys profile -o report ./myprogram     # custom output name
    $ nsys profile --stats=true ./myprogram  # print summary stats

    # Generate multiple output formats
    $ nsys profile -o report --export=sqlite,text ./myprogram

Common options:

.. code-block:: bash

    -o, --output=NAME           # output file name (without extension)
    -t, --trace=TRACE           # what to trace: cuda,nvtx,osrt,cublas,cudnn
    --stats=true                # print summary statistics
    --force-overwrite=true      # overwrite existing report
    -w, --show-output=true      # show application output
    --sample=cpu                # CPU sampling
    --cudabacktrace=true        # CUDA API backtraces

Trace specific APIs:

.. code-block:: bash

    $ nsys profile -t cuda,nvtx ./myprogram      # CUDA + NVTX markers
    $ nsys profile -t cuda,osrt ./myprogram      # CUDA + OS runtime
    $ nsys profile -t cuda,cublas ./myprogram    # CUDA + cuBLAS

Viewing Results
---------------

Open the report in the GUI for interactive timeline exploration:

.. code-block:: bash

    $ nsys-ui report.nsys-rep    # open in GUI
    $ nsys stats report.nsys-rep # command-line statistics

Export to different formats for custom analysis:

.. code-block:: bash

    $ nsys export -t sqlite report.nsys-rep     # SQLite database
    $ nsys export -t text report.nsys-rep       # text summary

The GUI timeline shows:

- CPU thread activity and call stacks
- CUDA API calls (cudaMalloc, cudaMemcpy, kernel launches)
- GPU kernel executions with duration
- Memory transfers between host and device
- NVTX ranges and markers

NVTX Annotations
----------------

NVTX (NVIDIA Tools Extension) lets you add custom markers and ranges to your
code, making the timeline easier to understand. Ranges show up as colored bars
in the timeline, helping you correlate application phases with GPU activity.

.. code-block:: cpp

    #include <nvtx3/nvToolsExt.h>

    void myFunction() {
      nvtxRangePush("myFunction");
      // ... work ...
      nvtxRangePop();
    }

    // Or with colors
    nvtxEventAttributes_t attr = {0};
    attr.version = NVTX_VERSION;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = 0xFF00FF00;  // green
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = "Important Section";
    nvtxRangePushEx(&attr);

Compile with:

.. code-block:: bash

    $ nvcc -o myprogram main.cu -lnvToolsExt

Common Workflows
----------------

**Profile and get summary statistics:**

.. code-block:: bash

    $ nsys profile --stats=true -o system ./myprogram
    # Shows kernel times, memory transfer times, API call counts

**Identify CPU-GPU synchronization issues:**

.. code-block:: bash

    $ nsys profile -t cuda ./myprogram
    $ nsys-ui system.nsys-rep
    # Look for gaps between kernel launches (CPU idle or sync points)

**Profile a running application:**

.. code-block:: bash

    $ nsys profile --attach-pid=1234 -o attached
    # Or launch with delayed start
    $ nsys profile --delay=5 ./myprogram   # start profiling after 5 seconds

**Profile specific duration:**

.. code-block:: bash

    $ nsys profile --duration=10 ./myprogram   # profile for 10 seconds

Remote Profiling
----------------

Profile applications running on remote machines or clusters where you can't
run the GUI directly:

.. code-block:: bash

    # On remote machine
    $ nsys profile -o /tmp/report ./myprogram

    # Copy report to local machine
    $ scp remote:/tmp/report.nsys-rep .
    $ nsys-ui report.nsys-rep

Nsight Systems GUI also supports connecting to remote machines via SSH for
interactive profiling sessions.

What to Look For
----------------

Common performance issues visible in the timeline:

.. code-block:: text

    Issue                       Timeline Pattern
    ─────────────────────────────────────────────────────────────────
    CPU-GPU serialization       Gaps between kernel launches
    Excessive synchronization   Many cudaDeviceSynchronize calls
    Memory transfer overhead    Large cudaMemcpy blocks
    Kernel launch overhead      Many small kernels with gaps
    Underutilized GPU           Long CPU sections, short GPU bursts
