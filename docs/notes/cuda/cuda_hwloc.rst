===================
Hardware Topology
===================

.. meta::
   :description: Hardware topology discovery using hwloc for GPU affinity, NUMA awareness, and optimal CPU-GPU-NIC placement.
   :keywords: hwloc, NUMA, GPU affinity, topology, PCI, EFA, NVLink, CPU pinning

.. contents:: Table of Contents
    :backlinks: none

Understanding hardware topology is critical for high-performance GPU applications.
Modern systems have complex NUMA architectures where memory access latency varies
based on which CPU socket accesses which memory region. Similarly, GPUs and network
devices (like EFA/InfiniBand) are connected through specific PCI bridges, and
placing communicating components on the same NUMA node minimizes latency.

The ``hwloc`` (Hardware Locality) library provides a portable abstraction for
discovering and querying hardware topology across different platforms. For
comprehensive GPU affinity examples including EFA device discovery and NVLink
bandwidth detection, see `Libefaxx <https://github.com/crazyguitar/Libefaxx>`_.
The hwloc project also provides `lstopo <https://github.com/open-mpi/hwloc/blob/master/utils/lstopo/lstopo.c>`_,
a reference implementation for topology visualization.

Basic Topology Discovery
------------------------

:Source: `src/cuda/hwloc-topology <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/hwloc-topology>`_

The hwloc library requires explicit initialization before use. The topology object
represents the entire hardware hierarchy from the machine level down to individual
cores and I/O devices. After calling ``hwloc_topology_init()``, the topology is
empty until ``hwloc_topology_load()`` performs the actual hardware discovery. The
depth value indicates how many levels exist in the hierarchy (machine, package,
NUMA node, L3 cache, core, etc.). Always destroy the topology when done to free
allocated resources.

.. code-block:: cpp

    #include <hwloc.h>

    hwloc_topology_t topology;
    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);

    int depth = hwloc_topology_get_depth(topology);
    printf("Topology depth: %d\n", depth);

    hwloc_topology_destroy(topology);

NUMA Node Discovery
-------------------

Non-Uniform Memory Access (NUMA) architectures have multiple memory controllers,
each associated with a subset of CPU cores. Accessing local memory (attached to
the same controller) is faster than accessing remote memory. For GPU workloads,
understanding NUMA topology helps place CPU threads on cores that have the lowest
latency path to both the GPU and its associated memory. The ``hwloc_get_type_depth()``
function returns the level in the hierarchy where NUMA nodes appear, and
``hwloc_get_nbobjs_by_depth()`` counts how many exist at that level.

.. code-block:: cpp

    hwloc_topology_t topology;
    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);

    int numa_depth = hwloc_get_type_depth(topology, HWLOC_OBJ_NUMANODE);
    int numa_count = hwloc_get_nbobjs_by_depth(topology, numa_depth);
    printf("NUMA nodes: %d\n", numa_count);

    for (int i = 0; i < numa_count; i++) {
      hwloc_obj_t numa = hwloc_get_obj_by_depth(topology, numa_depth, i);
      printf("NUMA %d: %lu MB local memory\n", i, numa->attr->numanode.local_memory / (1024 * 1024));
    }

    hwloc_topology_destroy(topology);

PCI Device Enumeration
----------------------

By default, hwloc only discovers CPU and memory topology. To find GPUs, network
adapters, and other I/O devices, you must enable I/O filtering before loading
the topology. The ``HWLOC_TYPE_FILTER_KEEP_IMPORTANT`` filter includes PCI devices
that have associated OS devices (like ``/dev/nvidia0`` or network interfaces),
filtering out bridges and other infrastructure. Each PCI device object contains
attributes including domain, bus, device, function numbers, and vendor/device IDs
that can be used to identify specific hardware.

.. code-block:: cpp

    hwloc_topology_t topology;
    hwloc_topology_init(&topology);
    hwloc_topology_set_io_types_filter(topology, HWLOC_TYPE_FILTER_KEEP_IMPORTANT);
    hwloc_topology_load(topology);

    hwloc_obj_t obj = nullptr;
    while ((obj = hwloc_get_next_pcidev(topology, obj)) != nullptr) {
      printf("PCI %04x:%02x:%02x.%x vendor=%04x device=%04x class=%04x\n",
             obj->attr->pcidev.domain,
             obj->attr->pcidev.bus,
             obj->attr->pcidev.dev,
             obj->attr->pcidev.func,
             obj->attr->pcidev.vendor_id,
             obj->attr->pcidev.device_id,
             obj->attr->pcidev.class_id);
    }

    hwloc_topology_destroy(topology);

GPU-to-CUDA Mapping
-------------------

CUDA assigns device indices (0, 1, 2, ...) that may not correspond to physical
slot order or PCI bus order. To correlate CUDA devices with hwloc topology, use
the PCI address from ``cudaDeviceProp`` (which provides ``pciDomainID``, ``pciBusID``,
and ``pciDeviceID``) to find the matching hwloc object. This mapping is essential
for determining which NUMA node a GPU belongs to, which CPU cores are local to it,
and which network adapters share the same PCI bridge for optimal data placement.

.. code-block:: cpp

    #include <cuda_runtime.h>
    #include <hwloc.h>

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    hwloc_topology_t topology;
    hwloc_topology_init(&topology);
    hwloc_topology_set_io_types_filter(topology, HWLOC_TYPE_FILTER_KEEP_IMPORTANT);
    hwloc_topology_load(topology);

    hwloc_obj_t obj = nullptr;
    while ((obj = hwloc_get_next_pcidev(topology, obj)) != nullptr) {
      if (obj->attr->pcidev.domain == prop.pciDomainID &&
          obj->attr->pcidev.bus == prop.pciBusID &&
          obj->attr->pcidev.dev == prop.pciDeviceID) {
        printf("Found GPU in hwloc: %04x:%02x:%02x\n",
               obj->attr->pcidev.domain,
               obj->attr->pcidev.bus,
               obj->attr->pcidev.dev);
        break;
      }
    }

    hwloc_topology_destroy(topology);

Identifying Device Types
------------------------

PCI devices are identified by their class ID and vendor ID. The class ID upper
byte indicates the device category: ``0x03`` for display controllers (GPUs),
``0x02`` for network controllers, ``0x01`` for storage. NVIDIA GPUs have vendor
ID ``0x10de``, AMD GPUs use ``0x1002``. Host bridges connect the CPU to PCI
hierarchies and are identified by their upstream type not being PCI. These
helper functions are useful when traversing topology to categorize devices.

.. code-block:: cpp

    constexpr uint16_t NVIDIA_VENDOR_ID = 0x10de;
    constexpr uint16_t AMD_VENDOR_ID = 0x1002;

    bool is_nvidia_gpu(hwloc_obj_t obj) {
      if (obj->type != HWLOC_OBJ_PCI_DEVICE) return false;
      auto class_id = obj->attr->pcidev.class_id >> 8;
      if (class_id != 0x03) return false;  // Display controller class
      return obj->attr->pcidev.vendor_id == NVIDIA_VENDOR_ID;
    }

    bool is_host_bridge(hwloc_obj_t obj) {
      if (obj->type != HWLOC_OBJ_BRIDGE) return false;
      return obj->attr->bridge.upstream_type != HWLOC_OBJ_BRIDGE_PCI;
    }

CPU Core Enumeration
--------------------

Each NUMA node has an associated cpuset bitmap indicating which CPU cores belong
to it. For GPU-intensive workloads, pinning application threads to cores on the
same NUMA node as the GPU reduces memory access latency and avoids cross-socket
traffic. The ``hwloc_bitmap_foreach_begin/end`` macros iterate over set bits in
the cpuset. This information can be used with ``pthread_setaffinity_np()`` or
``sched_setaffinity()`` to bind threads to specific cores.

.. code-block:: cpp

    hwloc_obj_t numa = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, 0);

    // Get cpuset for this NUMA node
    hwloc_cpuset_t cpuset = hwloc_bitmap_dup(numa->cpuset);

    int core_id;
    hwloc_bitmap_foreach_begin(core_id, cpuset) {
      printf("Core %d belongs to NUMA 0\n", core_id);
    }
    hwloc_bitmap_foreach_end();

    hwloc_bitmap_free(cpuset);

Topology Traversal
------------------

The hwloc topology is a tree structure with multiple child types: regular children
(CPU objects), memory children (NUMA nodes), I/O children (PCI devices), and misc
children. A complete traversal must visit all child types to build a full picture
of the hardware. This recursive approach is useful for building affinity maps that
associate GPUs with their local NUMA nodes, CPU cores, and network devices based
on their position in the PCI hierarchy.

.. code-block:: cpp

    void traverse(hwloc_obj_t obj, int depth) {
      for (int i = 0; i < depth; i++) printf("  ");
      printf("%s #%d\n", hwloc_obj_type_string(obj->type), obj->logical_index);

      // Traverse children
      for (hwloc_obj_t child = obj->first_child; child; child = child->next_sibling) {
        traverse(child, depth + 1);
      }
      // Traverse I/O children (PCI devices)
      for (hwloc_obj_t child = obj->io_first_child; child; child = child->next_sibling) {
        traverse(child, depth + 1);
      }
    }

    // Start from root
    traverse(hwloc_get_root_obj(topology), 0);
