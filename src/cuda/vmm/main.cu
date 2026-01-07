/**
 * CUDA Virtual Memory Management (VMM) Example
 *
 * VMM provides fine-grained control over GPU memory sharing. Unlike
 * cudaEnablePeerAccess which maps ALL allocations to peer devices,
 * VMM lets you choose exactly which allocations to share.
 *
 * VMM Flow:
 *   cuMemCreate ──▶ cuMemAddressReserve ──▶ cuMemMap ──▶ cuMemSetAccess
 *
 * Key APIs:
 *   cuMemCreate          - Allocate physical memory (returns handle, not pointer)
 *   cuMemAddressReserve  - Reserve virtual address range
 *   cuMemMap             - Map physical memory to virtual address
 *   cuMemSetAccess       - Set access permissions for devices
 *   cuMemUnmap           - Unmap memory
 *   cuMemRelease         - Release physical memory
 *   cuMemAddressFree     - Free virtual address range
 *
 * For IPC sharing:
 *   cuMemExportToShareableHandle   - Export handle for sharing
 *   cuMemImportFromShareableHandle - Import handle from another process
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

#define CU_CHECK(call)                                            \
  do {                                                            \
    CUresult err = (call);                                        \
    ASSERT_EQ(err, CUDA_SUCCESS) << "CUDA driver error: " << err; \
  } while (0)

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err); \
  } while (0)

__global__ void write_kernel(int* buf, int value, size_t len) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) buf[idx] = value * 1000 + idx;
}

__global__ void verify_kernel(int* buf, int expected, size_t len, int* errors) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    if (buf[idx] != expected * 1000 + (int)idx) atomicAdd(errors, 1);
  }
}

// Round up to alignment
static size_t roundUp(size_t size, size_t granularity) { return ((size + granularity - 1) / granularity) * granularity; }

/**
 * Basic VMM allocation and mapping on a single device.
 *
 * Flow:
 *   1. Query granularity requirements
 *   2. cuMemCreate - allocate physical memory
 *   3. cuMemAddressReserve - reserve virtual address range
 *   4. cuMemMap - map physical to virtual
 *   5. cuMemSetAccess - enable access for device
 *   6. Use memory in kernel
 *   7. Cleanup: cuMemUnmap, cuMemRelease, cuMemAddressFree
 */
TEST(VMM, BasicAllocation) {
  CU_CHECK(cuInit(0));

  CUdevice device;
  CU_CHECK(cuDeviceGet(&device, 0));

  // Check VMM support
  int vmmSupported = 0;
  CU_CHECK(cuDeviceGetAttribute(&vmmSupported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));
  if (!vmmSupported) {
    GTEST_SKIP() << "VMM not supported on this device";
  }

  CUcontext ctx;
  CU_CHECK(cuCtxCreate(&ctx, 0, device));

  // Setup allocation properties
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;

  // Get granularity
  size_t granularity = 0;
  CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  printf("Allocation granularity: %zu bytes\n", granularity);

  // Allocate physical memory
  size_t size = roundUp(1024 * sizeof(int), granularity);
  CUmemGenericAllocationHandle allocHandle;
  CU_CHECK(cuMemCreate(&allocHandle, size, &prop, 0));
  printf("Created physical allocation of %zu bytes\n", size);

  // Reserve virtual address range
  CUdeviceptr ptr = 0;
  CU_CHECK(cuMemAddressReserve(&ptr, size, granularity, 0, 0));
  printf("Reserved VA range at %p\n", (void*)ptr);

  // Map physical to virtual
  CU_CHECK(cuMemMap(ptr, size, 0, allocHandle, 0));
  printf("Mapped physical memory to VA\n");

  // Set access permissions
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = device;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CU_CHECK(cuMemSetAccess(ptr, size, &accessDesc, 1));
  printf("Set access permissions\n");

  // Use memory
  int* d_err;
  CUDA_CHECK(cudaMalloc(&d_err, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_err, 0, sizeof(int)));

  int block = 256;
  int grid = (1024 + block - 1) / block;
  write_kernel<<<grid, block>>>((int*)ptr, 42, 1024);
  verify_kernel<<<grid, block>>>((int*)ptr, 42, 1024, d_err);
  CUDA_CHECK(cudaDeviceSynchronize());

  int errors = 0;
  CUDA_CHECK(cudaMemcpy(&errors, d_err, sizeof(int), cudaMemcpyDeviceToHost));
  printf("Verification errors: %d\n", errors);
  EXPECT_EQ(errors, 0);

  // Cleanup
  CUDA_CHECK(cudaFree(d_err));
  CU_CHECK(cuMemUnmap(ptr, size));
  CU_CHECK(cuMemRelease(allocHandle));
  CU_CHECK(cuMemAddressFree(ptr, size));
  CU_CHECK(cuCtxDestroy(ctx));
}

/**
 * VMM with P2P access - share specific allocation with peer GPU.
 *
 * Unlike cudaEnablePeerAccess which maps ALL allocations,
 * VMM lets you share only specific allocations.
 */
TEST(VMM, P2PAccess) {
  CU_CHECK(cuInit(0));

  int deviceCount;
  CU_CHECK(cuDeviceGetCount(&deviceCount));
  if (deviceCount < 2) {
    GTEST_SKIP() << "Need at least 2 GPUs for P2P VMM test";
  }

  CUdevice dev0, dev1;
  CU_CHECK(cuDeviceGet(&dev0, 0));
  CU_CHECK(cuDeviceGet(&dev1, 1));

  // Check VMM support on both devices
  int vmm0 = 0, vmm1 = 0;
  CU_CHECK(cuDeviceGetAttribute(&vmm0, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev0));
  CU_CHECK(cuDeviceGetAttribute(&vmm1, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev1));
  if (!vmm0 || !vmm1) {
    GTEST_SKIP() << "VMM not supported on all devices";
  }

  CUcontext ctx0, ctx1;
  CU_CHECK(cuCtxCreate(&ctx0, 0, dev0));
  CU_CHECK(cuCtxCreate(&ctx1, 0, dev1));

  // Allocate on GPU 0
  CU_CHECK(cuCtxSetCurrent(ctx0));

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = dev0;

  size_t granularity = 0;
  CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

  size_t size = roundUp(1024 * sizeof(int), granularity);
  CUmemGenericAllocationHandle allocHandle;
  CU_CHECK(cuMemCreate(&allocHandle, size, &prop, 0));

  CUdeviceptr ptr = 0;
  CU_CHECK(cuMemAddressReserve(&ptr, size, granularity, 0, 0));
  CU_CHECK(cuMemMap(ptr, size, 0, allocHandle, 0));

  // Set access for BOTH GPU 0 and GPU 1
  CUmemAccessDesc accessDescs[2] = {};
  accessDescs[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDescs[0].location.id = dev0;
  accessDescs[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  accessDescs[1].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDescs[1].location.id = dev1;
  accessDescs[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CU_CHECK(cuMemSetAccess(ptr, size, accessDescs, 2));

  printf("GPU 0 allocation accessible from GPU 1 via VMM\n");

  // GPU 1 writes to GPU 0's memory
  CU_CHECK(cuCtxSetCurrent(ctx1));
  int block = 256;
  int grid = (1024 + block - 1) / block;
  write_kernel<<<grid, block>>>((int*)ptr, 99, 1024);
  CUDA_CHECK(cudaDeviceSynchronize());

  // GPU 0 verifies
  CU_CHECK(cuCtxSetCurrent(ctx0));
  int* d_err;
  CUDA_CHECK(cudaMalloc(&d_err, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_err, 0, sizeof(int)));
  verify_kernel<<<grid, block>>>((int*)ptr, 99, 1024, d_err);
  CUDA_CHECK(cudaDeviceSynchronize());

  int errors = 0;
  CUDA_CHECK(cudaMemcpy(&errors, d_err, sizeof(int), cudaMemcpyDeviceToHost));
  printf("P2P VMM verification errors: %d\n", errors);
  EXPECT_EQ(errors, 0);

  // Cleanup
  CUDA_CHECK(cudaFree(d_err));
  CU_CHECK(cuMemUnmap(ptr, size));
  CU_CHECK(cuMemRelease(allocHandle));
  CU_CHECK(cuMemAddressFree(ptr, size));
  CU_CHECK(cuCtxDestroy(ctx1));
  CU_CHECK(cuCtxDestroy(ctx0));
}
