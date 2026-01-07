========
Hardware
========

.. meta::
   :description: Linux hardware information and management commands covering CPU, memory, disk, and PCI device diagnostics.
   :keywords: Linux hardware, lscpu, lspci, disk management, system information, hardware diagnostics

.. contents:: Table of Contents
    :backlinks: none

Understanding your system's hardware is crucial for performance tuning and troubleshooting.
These commands reveal CPU specifications, memory configuration, disk health, and PCI device
details—information essential for capacity planning and diagnosing hardware issues.

An interactive demo script is available at `src/bash/hardware.sh <https://github.com/crazyguitar/cppcheatsheet/blob/master/src/bash/hardware.sh>`_
to help you experiment with the concepts covered in this cheatsheet.

.. code-block:: bash

    ./src/bash/hardware.sh           # Run all demos
    ./src/bash/hardware.sh cpu       # Run CPU info demo
    ./src/bash/hardware.sh disk      # Run disk info demo
    ./src/bash/hardware.sh --help    # Show all available sections

CPU Information
===============

Commands for viewing processor details, architecture, and performance characteristics.

Basic CPU Info
--------------

.. code-block:: bash

    nproc                               # Number of processing units
    nproc --all                         # All available processors
    lscpu                               # Detailed CPU architecture info
    lscpu -e                            # Extended CPU info table
    lscpu -p                            # Parseable output

    # macOS
    sysctl -n hw.logicalcpu             # Logical CPU count
    sysctl -n hw.physicalcpu            # Physical core count
    sysctl -a | grep machdep.cpu        # Detailed CPU info

Detailed CPU Information
------------------------

.. code-block:: bash

    cat /proc/cpuinfo                   # Raw CPU info
    cat /proc/cpuinfo | grep "model name" | head -1
    cat /proc/cpuinfo | grep "cpu MHz"  # Current frequencies
    cat /proc/cpuinfo | grep "cache size"

    # CPU flags/features
    cat /proc/cpuinfo | grep flags | head -1

    # Count physical/logical CPUs
    grep -c ^processor /proc/cpuinfo    # Logical CPUs
    grep "physical id" /proc/cpuinfo | sort -u | wc -l  # Physical CPUs
    grep "cpu cores" /proc/cpuinfo | head -1            # Cores per CPU

CPU Performance
---------------

.. code-block:: bash

    # CPU frequency scaling
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
    cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

    # CPU usage
    mpstat -P ALL 1                     # Per-CPU stats (sysstat)
    sar -u 1 5                          # CPU utilization over time

Memory Information
==================

Commands for viewing RAM usage, configuration, and hardware details.

Memory Usage
------------

.. code-block:: bash

    free -h                             # Human-readable memory usage
    free -m                             # Memory in MB
    free -g                             # Memory in GB
    free -s 2                           # Update every 2 seconds

    # Detailed memory info
    cat /proc/meminfo
    cat /proc/meminfo | grep -E "MemTotal|MemFree|MemAvailable|Buffers|Cached"

    # macOS
    vm_stat                             # Virtual memory stats
    top -l 1 | head -n 10               # Memory summary

Memory Hardware
---------------

.. code-block:: bash

    # Physical memory details (requires root)
    dmidecode -t memory                 # Memory module info
    dmidecode -t memory | grep -E "Size|Type|Speed|Manufacturer"

    # Memory slots
    dmidecode -t memory | grep -A 16 "Memory Device" | grep -E "Size|Locator|Type:"

    # Numa topology
    numactl --hardware                  # NUMA node info
    lsmem                               # Memory ranges

Disk and Storage
================

Commands for disk space, partitions, and storage device management.

Disk Space Usage
----------------

.. code-block:: bash

    df -h                               # Disk space (human-readable)
    df -h /                             # Specific mount point
    df -hT                              # Include filesystem type
    df -i                               # Inode usage

    # Directory sizes
    du -sh /path                        # Single directory size
    du -sh /path/*                      # Size of subdirectories
    du -sh /path/* | sort -h            # Sorted by size
    du -sh /path/* | sort -rh | head -10  # Top 10 largest
    du -d 1 -h /path                    # Depth-limited

    # Find large files
    find / -type f -size +100M -exec ls -lh {} \; 2>/dev/null
    du -ah /path | sort -rh | head -20  # Largest files/dirs

Block Devices
-------------

.. code-block:: bash

    lsblk                               # List block devices
    lsblk -f                            # Include filesystem info
    lsblk -o NAME,SIZE,TYPE,MOUNTPOINT,FSTYPE
    lsblk -d -o NAME,SIZE,MODEL,SERIAL  # Disk details

    blkid                               # Block device attributes
    blkid /dev/sda1                     # Specific device

Partition Management
--------------------

.. code-block:: bash

    fdisk -l                            # List partitions (MBR)
    fdisk -l /dev/sda                   # Specific disk
    gdisk -l /dev/sda                   # GPT partitions
    parted -l                           # All partition tables

    # Partition info
    cat /proc/partitions
    sfdisk -l /dev/sda                  # Scriptable fdisk

Filesystem Operations
---------------------

.. code-block:: bash

    # Mount/unmount
    mount /dev/sda1 /mnt                # Mount device
    umount /mnt                         # Unmount
    mount -o remount,rw /               # Remount with options

    # Filesystem info
    tune2fs -l /dev/sda1                # ext filesystem details
    xfs_info /dev/sda1                  # XFS filesystem info
    dumpe2fs /dev/sda1 | head -50       # ext superblock info

    # Filesystem check
    fsck /dev/sda1                      # Check filesystem
    e2fsck -f /dev/sda1                 # Force check ext

Disk Health and Performance
---------------------------

.. code-block:: bash

    # SMART monitoring
    smartctl -a /dev/sda                # All SMART data
    smartctl -H /dev/sda                # Health status
    smartctl -t short /dev/sda          # Run short test
    smartctl -l selftest /dev/sda       # Test results

    # I/O statistics
    iostat -x 1                         # Extended I/O stats
    iostat -d -x /dev/sda 1             # Specific device
    iotop                               # I/O by process

    # Disk benchmark
    hdparm -Tt /dev/sda                 # Read speed test
    dd if=/dev/zero of=test bs=1G count=1 oflag=direct  # Write test

PCI Devices
===========

The ``lspci`` command lists all PCI devices including network cards, GPUs, storage
controllers, and other hardware components.

Basic Usage
-----------

.. code-block:: bash

    lspci                               # List all PCI devices
    lspci -v                            # Verbose output
    lspci -vv                           # Very verbose
    lspci -vvv                          # Maximum verbosity

    lspci -k                            # Show kernel drivers
    lspci -nn                           # Show numeric IDs
    lspci -mm                           # Machine-readable output

Filtering Devices
-----------------

.. code-block:: bash

    # By device class
    lspci | grep -i vga                 # Graphics cards
    lspci | grep -i network             # Network devices
    lspci | grep -i audio               # Audio devices
    lspci | grep -i usb                 # USB controllers
    lspci | grep -i sata                # SATA controllers
    lspci | grep -i nvme                # NVMe controllers

    # By vendor
    lspci -d 10de:                      # NVIDIA devices (vendor 10de)
    lspci -d 8086:                      # Intel devices
    lspci -d 1002:                      # AMD devices

    # By slot
    lspci -s 00:02.0                    # Specific slot
    lspci -s 00:02.0 -v                 # Detailed info for slot

Detailed Device Information
---------------------------

.. code-block:: bash

    # Show device tree
    lspci -t                            # Tree view
    lspci -tv                           # Tree with details

    # Kernel modules
    lspci -k | grep -A 3 "VGA"          # GPU driver info
    lspci -v -s 00:02.0 | grep "Kernel"

    # Device capabilities
    lspci -vvv -s 00:1f.0 | grep -i capabilities

    # Update PCI database
    update-pciids                       # Download latest IDs
