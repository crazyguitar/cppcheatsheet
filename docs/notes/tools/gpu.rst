===
GPU
===

.. meta::
   :description: GPU monitoring and management commands covering nvidia-smi, CUDA, multi-GPU systems, and graphics card diagnostics on Linux.
   :keywords: nvidia-smi, GPU monitoring, CUDA, multi-GPU, NVLink, MIG, graphics card, Linux GPU, deep learning

.. contents:: Table of Contents
    :backlinks: none

For deep learning, scientific computing, and graphics workloads, nvidia-smi is the
essential tool for GPU monitoring and management. From checking memory usage to tuning
power limits and managing multi-GPU configurations, these commands help maximize GPU
utilization and diagnose performance bottlenecks.

An interactive demo script is available at `src/bash/gpu.sh <https://github.com/crazyguitar/cppcheatsheet/blob/master/src/bash/gpu.sh>`_
to help you experiment with the concepts covered in this cheatsheet.

.. code-block:: bash

    ./src/bash/gpu.sh           # Run all demos
    ./src/bash/gpu.sh query     # Run query format demo
    ./src/bash/gpu.sh --help    # Show all available sections

Basic GPU Information
=====================

Commands for viewing graphics card details and driver information.

.. code-block:: bash

    lspci | grep -i vga                 # List graphics cards
    lspci | grep -i nvidia              # NVIDIA devices
    lspci -v -s $(lspci | grep -i vga | cut -d' ' -f1)  # Detailed GPU info

    # OpenGL info
    glxinfo | grep -i "renderer\|vendor\|version"

    # Vulkan info
    vulkaninfo --summary

nvidia-smi Basics
=================

The nvidia-smi tool provides monitoring and management capabilities for NVIDIA GPUs.

.. code-block:: bash

    nvidia-smi                          # GPU status overview
    nvidia-smi -L                       # List GPUs
    nvidia-smi -q                       # Full query (all details)
    nvidia-smi -q -d MEMORY             # Memory details
    nvidia-smi -q -d UTILIZATION        # Utilization details
    nvidia-smi -q -d TEMPERATURE        # Temperature details
    nvidia-smi -q -d POWER              # Power details
    nvidia-smi -q -d CLOCK              # Clock speeds

nvidia-smi Monitoring
=====================

Commands for continuous GPU monitoring and process tracking.

.. code-block:: bash

    # Continuous monitoring
    nvidia-smi -l 1                     # Update every 1 second
    nvidia-smi dmon -s u                # Device monitoring (utilization)
    nvidia-smi dmon -s p                # Power monitoring
    nvidia-smi dmon -s t                # Temperature monitoring
    nvidia-smi dmon -s m                # Memory monitoring

    # Process monitoring
    nvidia-smi pmon -s u                # Process utilization
    nvidia-smi pmon -s m                # Process memory

    # Watch specific metrics
    watch -n 1 nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv

nvidia-smi Query Format
=======================

Custom queries allow extracting specific GPU metrics in various formats.

.. code-block:: bash

    # Custom query output
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
    nvidia-smi --query-gpu=gpu_name,gpu_bus_id,vbios_version --format=csv
    nvidia-smi --query-gpu=temperature.gpu,fan.speed,power.draw,power.limit --format=csv
    nvidia-smi --query-gpu=clocks.gr,clocks.mem,clocks.sm --format=csv

    # Query specific GPU
    nvidia-smi -i 0 --query-gpu=name,memory.used --format=csv
    nvidia-smi -i 0,1 --query-gpu=name,utilization.gpu --format=csv

    # Process queries
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

    # Available query options
    nvidia-smi --help-query-gpu         # List all GPU query options
    nvidia-smi --help-query-compute-apps  # List process query options

Common Query Fields
-------------------

.. code-block:: bash

    # GPU identification
    name, gpu_name, gpu_bus_id, gpu_serial, gpu_uuid, vbios_version

    # Memory
    memory.total, memory.used, memory.free

    # Utilization
    utilization.gpu, utilization.memory

    # Temperature and power
    temperature.gpu, fan.speed, power.draw, power.limit

    # Clocks
    clocks.gr, clocks.mem, clocks.sm, clocks.max.gr, clocks.max.mem

    # Performance
    pstate, performance_state

nvidia-smi Management
=====================

Commands for configuring GPU settings and performance tuning.

.. code-block:: bash

    # Set persistence mode (reduces latency)
    nvidia-smi -pm 1                    # Enable
    nvidia-smi -pm 0                    # Disable

    # Set power limit (watts)
    nvidia-smi -pl 250                  # Set to 250W

    # Set GPU clocks
    nvidia-smi -lgc 1200,1800           # Lock graphics clock range
    nvidia-smi -rgc                     # Reset graphics clocks

    # Set compute mode
    nvidia-smi -c 0                     # Default
    nvidia-smi -c 1                     # Exclusive thread
    nvidia-smi -c 2                     # Prohibited
    nvidia-smi -c 3                     # Exclusive process

    # Reset GPU
    nvidia-smi -r                       # Reset GPU

    # Enable/disable ECC
    nvidia-smi -e 1                     # Enable ECC
    nvidia-smi -e 0                     # Disable ECC

Multi-GPU Systems
=================

Commands for managing multiple GPUs, topology, and interconnects.

.. code-block:: bash

    # Topology
    nvidia-smi topo -m                  # GPU topology matrix
    nvidia-smi topo -p                  # P2P capabilities

    # NVLink status
    nvidia-smi nvlink -s               # NVLink status
    nvidia-smi nvlink -c               # NVLink capabilities

    # MIG (Multi-Instance GPU) - A100/H100
    nvidia-smi mig -lgip               # List GPU instance profiles
    nvidia-smi mig -cgi 9,9            # Create GPU instances
    nvidia-smi mig -dci                # Destroy compute instances

CUDA Environment
================

Environment variables and commands for CUDA configuration.

.. code-block:: bash

    # Check CUDA version
    nvcc --version
    cat /usr/local/cuda/version.txt

    # CUDA environment variables
    export CUDA_VISIBLE_DEVICES=0       # Use only GPU 0
    export CUDA_VISIBLE_DEVICES=0,1     # Use GPUs 0 and 1
    export CUDA_VISIBLE_DEVICES=""      # Hide all GPUs

    # Check CUDA device properties
    nvidia-smi -q -d COMPUTE

    # cuDNN version
    cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
