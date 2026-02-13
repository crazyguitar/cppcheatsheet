// NCCL AllGather
//
// AllGather concatenates each rank's buffer into a single output on every
// rank. Each rank contributes sendcount elements and receives
// sendcount * nRanks elements.
//
// Before:  rank0=[A]    rank1=[B]
// After:   rank0=[A,B]  rank1=[A,B]

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

struct AllGather {
  const int ndev;
  std::vector<ncclComm_t> comms;
  std::vector<cudaStream_t> streams;

  AllGather(int ndev) : ndev{ndev}, comms(ndev), streams(ndev) {
    std::vector<int> devs(ndev);
    for (int i = 0; i < ndev; i++) devs[i] = i;
    NCCL_CHECK(ncclCommInitAll(comms.data(), ndev, devs.data()));
    for (int i = 0; i < ndev; i++) {
      cudaSetDevice(i);
      cudaStreamCreate(&streams[i]);
    }
  }

  ~AllGather() {
    for (int i = 0; i < ndev; i++) {
      cudaSetDevice(i);
      cudaStreamDestroy(streams[i]);
      ncclCommDestroy(comms[i]);
    }
  }

  void operator()(Buffer& send, Buffer& recv) {
    NCCL_GROUP({
      for (int i = 0; i < ndev; i++) {
        NCCL_CHECK(ncclAllGather(send[i], recv[i], send.count, ncclFloat, comms[i], streams[i]));
      }
    });
    for (int i = 0; i < ndev; i++) {
      cudaSetDevice(i);
      cudaStreamSynchronize(streams[i]);
    }
  }
};

TEST(NCCL, AllGather) {
  int nDev = 0;
  cudaGetDeviceCount(&nDev);
  if (nDev < 2) GTEST_SKIP() << "Need >= 2 GPUs";

  constexpr int sendcount = 512;
  const int recvcount = sendcount * nDev;
  Buffer send(nDev, sendcount), recv(nDev, recvcount);
  for (int i = 0; i < nDev; i++) send.fill(i, static_cast<float>(i));

  AllGather ag(nDev);
  ag(send, recv);

  for (int i = 0; i < nDev; i++) {
    std::vector<float> h(recvcount);
    cudaSetDevice(i);
    cudaMemcpy(h.data(), recv[i], recvcount * sizeof(float), cudaMemcpyDeviceToHost);
    for (int r = 0; r < nDev; r++) {
      EXPECT_FLOAT_EQ(h[r * sendcount], static_cast<float>(r)) << "rank " << i << " chunk " << r;
    }
  }
}
