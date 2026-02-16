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
#include <memory>
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

  void reset() {
    for (int i = 0; i < ndev; i++) {
      cudaSetDevice(i);
      cudaMemset(d_buf[i], 0, size);
    }
  }
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

// NCCL AlltoAll with CUDA Graph Stream Capture
//
// Grouped Send/Recv calls can also be captured into a CUDA graph.
// The entire ncclGroupStart/ncclGroupEnd block is captured as a single
// graph node.

struct CUDAGraph {
  cudaGraphExec_t exec_ = nullptr;

  template <typename F>
  CUDAGraph(cudaStream_t s, F&& fn) {
    cudaGraph_t g;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    fn();
    cudaStreamEndCapture(s, &g);
    cudaGraphInstantiate(&exec_, g, nullptr, nullptr, 0);
    cudaGraphDestroy(g);
  }

  ~CUDAGraph() {
    if (exec_) cudaGraphExecDestroy(exec_);
  }
  CUDAGraph(const CUDAGraph&) = delete;
  CUDAGraph& operator=(const CUDAGraph&) = delete;

  void launch(cudaStream_t s) { cudaGraphLaunch(exec_, s); }

  static void Launch(std::vector<std::unique_ptr<CUDAGraph>>& gs, std::vector<cudaStream_t>& streams, int nDev) {
    for (int i = 0; i < nDev; i++) {
      cudaSetDevice(i);
      gs[i]->launch(streams[i]);
    }
  }

  static void Sync(std::vector<cudaStream_t>& streams, int nDev) {
    for (int i = 0; i < nDev; i++) {
      cudaSetDevice(i);
      cudaStreamSynchronize(streams[i]);
    }
  }
};

TEST(NCCL, AlltoAllGraphCapture) {
  int nDev = 0;
  cudaGetDeviceCount(&nDev);
  if (nDev < 2) GTEST_SKIP() << "Need >= 2 GPUs";

  constexpr int chunkCount = 512;
  const int totalCount = chunkCount * nDev;
  AlltoAll a2a(nDev);

  Buffer send(nDev, totalCount), recv(nDev, totalCount);
  for (int i = 0; i < nDev; i++) {
    std::vector<float> h(totalCount);
    for (int peer = 0; peer < nDev; peer++) std::fill_n(h.data() + peer * chunkCount, chunkCount, static_cast<float>(i * nDev + peer));
    cudaSetDevice(i);
    cudaMemcpy(send[i], h.data(), totalCount * sizeof(float), cudaMemcpyHostToDevice);
  }

  // Warmup
  a2a(send, recv, chunkCount);

  // Capture
  std::vector<std::unique_ptr<CUDAGraph>> graphs(nDev);
  for (int i = 0; i < nDev; i++) {
    cudaSetDevice(i);
    graphs[i] = std::make_unique<CUDAGraph>(a2a.streams[i], [&, i] {
      NCCL_CHECK(ncclGroupStart());
      for (int peer = 0; peer < nDev; peer++) {
        NCCL_CHECK(ncclSend(send[i] + peer * chunkCount, chunkCount, ncclFloat, peer, a2a.comms[i], a2a.streams[i]));
        NCCL_CHECK(ncclRecv(recv[i] + peer * chunkCount, chunkCount, ncclFloat, peer, a2a.comms[i], a2a.streams[i]));
      }
      NCCL_CHECK(ncclGroupEnd());
    });
  }

  // Reset and replay
  recv.reset();
  CUDAGraph::Launch(graphs, a2a.streams, nDev);
  CUDAGraph::Sync(a2a.streams, nDev);

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
