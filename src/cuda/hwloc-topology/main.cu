#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <hwloc.h>

TEST(Hwloc, TopologyInit) {
  hwloc_topology_t topology;
  ASSERT_EQ(hwloc_topology_init(&topology), 0);
  ASSERT_EQ(hwloc_topology_load(topology), 0);

  int depth = hwloc_topology_get_depth(topology);
  EXPECT_GT(depth, 0);

  hwloc_topology_destroy(topology);
}

TEST(Hwloc, NumaNodes) {
  hwloc_topology_t topology;
  hwloc_topology_init(&topology);
  hwloc_topology_load(topology);

  int numa_depth = hwloc_get_type_depth(topology, HWLOC_OBJ_NUMANODE);
  if (numa_depth != HWLOC_TYPE_DEPTH_UNKNOWN) {
    int numa_count = hwloc_get_nbobjs_by_depth(topology, numa_depth);
    EXPECT_GE(numa_count, 0);
  }

  hwloc_topology_destroy(topology);
}

TEST(Hwloc, PCIDevices) {
  hwloc_topology_t topology;
  hwloc_topology_init(&topology);
  hwloc_topology_set_io_types_filter(topology, HWLOC_TYPE_FILTER_KEEP_IMPORTANT);
  hwloc_topology_load(topology);

  hwloc_obj_t obj = nullptr;
  while ((obj = hwloc_get_next_pcidev(topology, obj)) != nullptr) {
    EXPECT_EQ(obj->type, HWLOC_OBJ_PCI_DEVICE);
  }

  hwloc_topology_destroy(topology);
}

TEST(Hwloc, GPUAffinity) {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) GTEST_SKIP() << "No CUDA devices";

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  hwloc_topology_t topology;
  hwloc_topology_init(&topology);
  hwloc_topology_set_io_types_filter(topology, HWLOC_TYPE_FILTER_KEEP_IMPORTANT);
  hwloc_topology_load(topology);

  // Find GPU by PCI address
  hwloc_obj_t obj = nullptr;
  bool found = false;
  while ((obj = hwloc_get_next_pcidev(topology, obj)) != nullptr) {
    if (obj->attr->pcidev.domain == prop.pciDomainID &&
        obj->attr->pcidev.bus == prop.pciBusID &&
        obj->attr->pcidev.dev == prop.pciDeviceID) {
      found = true;
      break;
    }
  }

  hwloc_topology_destroy(topology);
  EXPECT_TRUE(found) << "GPU not found in hwloc topology";
}
