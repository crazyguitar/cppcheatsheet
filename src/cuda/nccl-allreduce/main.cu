// NCCL AllReduce
//
// AllReduce combines values from all ranks using a reduction operation (sum,
// prod, max, min) and distributes the result back to every rank. This is the
// most common collective in distributed deep learning for gradient
// synchronization.
//
// Before:  rank0=[1,2,3]  rank1=[4,5,6]
// After:   rank0=[5,7,9]  rank1=[5,7,9]   (element-wise sum)

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <nccl.h>

#include <cassert>
#include <vector>

#define NCCL_CHECK(cmd)                                \
  do {                                                 \
    ncclResult_t r = cmd;                              \
    assert(r == ncclSuccess && ncclGetErrorString(r)); \
  } while (0)

#define NCCL_GROUP(body)        \
  NCCL_CHECK(ncclGroupStart()); \
  body NCCL_CHECK(ncclGroupEnd())

struct Buffer {
  const int ndev;
  const int count;
  const size_t size;
  std::vector<float*> d_buf;

  Buffer(int ndev, int count) : ndev{ndev}, count{count}, size{count * sizeof(float)}, d_buf(ndev) {
    for (int i = 0; i < ndev; i++) {
      cudaSetDevice(i);
      cudaMalloc(&d_buf[i], size);
    }
  }

  ~Buffer() {
    for (int i = 0; i < ndev; i++) {
      cudaSetDevice(i);
      cudaFree(d_buf[i]);
    }
  }

  void fill(int dev, float val) {
    std::vector<float> h(count, val);
    cudaSetDevice(dev);
    cudaMemcpy(d_buf[dev], h.data(), size, cudaMemcpyHostToDevice);
  }

  float* operator[](int i) { return d_buf[i]; }
};

struct AllReduce {
  const int ndev;
  std::vector<ncclComm_t> comms;
  std::vector<cudaStream_t> streams;

  AllReduce(int ndev) : ndev{ndev}, comms(ndev), streams(ndev) {
    std::vector<int> devs(ndev);
    for (int i = 0; i < ndev; i++) devs[i] = i;
    NCCL_CHECK(ncclCommInitAll(comms.data(), ndev, devs.data()));
    for (int i = 0; i < ndev; i++) {
      cudaSetDevice(i);
      cudaStreamCreate(&streams[i]);
    }
  }

  ~AllReduce() {
    for (int i = 0; i < ndev; i++) {
      cudaSetDevice(i);
      cudaStreamDestroy(streams[i]);
      ncclCommDestroy(comms[i]);
    }
  }

  void operator()(Buffer& send, Buffer& recv) {
    NCCL_GROUP({
      for (int i = 0; i < ndev; i++) {
        NCCL_CHECK(ncclAllReduce(send[i], recv[i], send.count, ncclFloat, ncclSum, comms[i], streams[i]));
      }
    });
    for (int i = 0; i < ndev; i++) {
      cudaSetDevice(i);
      cudaStreamSynchronize(streams[i]);
    }
  }
};

TEST(NCCL, AllReduce) {
  int nDev = 0;
  cudaGetDeviceCount(&nDev);
  if (nDev < 2) GTEST_SKIP() << "Need >= 2 GPUs";

  constexpr int count = 1024;
  Buffer send(nDev, count), recv(nDev, count);
  for (int i = 0; i < nDev; i++) send.fill(i, static_cast<float>(i + 1));

  AllReduce ar(nDev);
  ar(send, recv);

  // Expected: 1 + 2 + ... + nDev
  float expected = static_cast<float>(nDev * (nDev + 1) / 2);
  for (int i = 0; i < nDev; i++) {
    std::vector<float> h(count);
    cudaSetDevice(i);
    cudaMemcpy(h.data(), recv[i], count * sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_FLOAT_EQ(h[0], expected) << "rank " << i;
    EXPECT_FLOAT_EQ(h[count - 1], expected) << "rank " << i;
  }
}
