// NCCL AlltoAll (naive via grouped Send/Recv)
//
// NCCL has no dedicated AlltoAll primitive. It can be composed by grouping
// point-to-point Send/Recv calls. Each rank sends a distinct chunk to every
// other rank.
//
// Before:  rank0=[A0,A1]  rank1=[B0,B1]   (2 ranks, chunk per peer)
// After:   rank0=[A0,B0]  rank1=[A1,B1]

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

struct AlltoAll {
  const int ndev;
  std::vector<ncclComm_t> comms;
  std::vector<cudaStream_t> streams;

  AlltoAll(int ndev) : ndev{ndev}, comms(ndev), streams(ndev) {
    std::vector<int> devs(ndev);
    for (int i = 0; i < ndev; i++) devs[i] = i;
    NCCL_CHECK(ncclCommInitAll(comms.data(), ndev, devs.data()));
    for (int i = 0; i < ndev; i++) {
      cudaSetDevice(i);
      cudaStreamCreate(&streams[i]);
    }
  }

  ~AlltoAll() {
    for (int i = 0; i < ndev; i++) {
      cudaSetDevice(i);
      cudaStreamDestroy(streams[i]);
      ncclCommDestroy(comms[i]);
    }
  }

  void operator()(Buffer& send, Buffer& recv, int chunkCount) {
    NCCL_GROUP({
      for (int i = 0; i < ndev; i++) {
        for (int peer = 0; peer < ndev; peer++) {
          NCCL_CHECK(ncclSend(send[i] + peer * chunkCount, chunkCount, ncclFloat, peer, comms[i], streams[i]));
          NCCL_CHECK(ncclRecv(recv[i] + peer * chunkCount, chunkCount, ncclFloat, peer, comms[i], streams[i]));
        }
      }
    });
    for (int i = 0; i < ndev; i++) {
      cudaSetDevice(i);
      cudaStreamSynchronize(streams[i]);
    }
  }
};

TEST(NCCL, AlltoAll) {
  int nDev = 0;
  cudaGetDeviceCount(&nDev);
  if (nDev < 2) GTEST_SKIP() << "Need >= 2 GPUs";

  constexpr int chunkCount = 512;
  const int totalCount = chunkCount * nDev;

  // send[rank] = [chunk_for_rank0, chunk_for_rank1, ...]
  // Each chunk filled with (rank * nDev + peer) so values are unique
  Buffer send(nDev, totalCount), recv(nDev, totalCount);
  for (int i = 0; i < nDev; i++) {
    std::vector<float> h(totalCount);
    for (int peer = 0; peer < nDev; peer++) {
      float val = static_cast<float>(i * nDev + peer);
      std::fill_n(h.data() + peer * chunkCount, chunkCount, val);
    }
    cudaSetDevice(i);
    cudaMemcpy(send[i], h.data(), totalCount * sizeof(float), cudaMemcpyHostToDevice);
  }

  AlltoAll a2a(nDev);
  a2a(send, recv, chunkCount);

  // recv[rank][peer_chunk] should contain value (peer * nDev + rank)
  for (int i = 0; i < nDev; i++) {
    std::vector<float> h(totalCount);
    cudaSetDevice(i);
    cudaMemcpy(h.data(), recv[i], totalCount * sizeof(float), cudaMemcpyDeviceToHost);
    for (int peer = 0; peer < nDev; peer++) {
      float expected = static_cast<float>(peer * nDev + i);
      EXPECT_FLOAT_EQ(h[peer * chunkCount], expected) << "rank " << i << " from peer " << peer;
    }
  }
}
