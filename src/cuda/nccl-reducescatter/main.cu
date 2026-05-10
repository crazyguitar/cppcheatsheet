// NCCL ReduceScatter
//
// ReduceScatter reduces across ranks then scatters equal chunks to each rank.
// Each rank ends up with a reduced slice of the data.
//
// Before:  rank0=[1,2,3,4]  rank1=[5,6,7,8]  (2 ranks, 4 elements)
// After:   rank0=[6,8]      rank1=[10,12]     (sum, 2 elements each)

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

struct ReduceScatter {
  const int ndev;
  std::vector<ncclComm_t> comms;
  std::vector<cudaStream_t> streams;

  ReduceScatter(int ndev) : ndev{ndev}, comms(ndev), streams(ndev) {
    std::vector<int> devs(ndev);
    for (int i = 0; i < ndev; i++) devs[i] = i;
    NCCL_CHECK(ncclCommInitAll(comms.data(), ndev, devs.data()));
    for (int i = 0; i < ndev; i++) {
      cudaSetDevice(i);
      cudaStreamCreate(&streams[i]);
    }
  }

  ~ReduceScatter() {
    for (int i = 0; i < ndev; i++) {
      cudaSetDevice(i);
      cudaStreamDestroy(streams[i]);
      ncclCommDestroy(comms[i]);
    }
  }

  void operator()(Buffer& send, Buffer& recv) {
    NCCL_GROUP({
      for (int i = 0; i < ndev; i++) {
        NCCL_CHECK(ncclReduceScatter(send[i], recv[i], recv.count, ncclFloat, ncclSum, comms[i], streams[i]));
      }
    });
    for (int i = 0; i < ndev; i++) {
      cudaSetDevice(i);
      cudaStreamSynchronize(streams[i]);
    }
  }
};

TEST(NCCL, ReduceScatter) {
  int nDev = 0;
  cudaGetDeviceCount(&nDev);
  if (nDev < 2) GTEST_SKIP() << "Need >= 2 GPUs";

  constexpr int recvcount = 512;
  const int sendcount = recvcount * nDev;
  Buffer send(nDev, sendcount), recv(nDev, recvcount);
  for (int i = 0; i < nDev; i++) send.fill(i, static_cast<float>(i + 1));

  ReduceScatter rs(nDev);
  rs(send, recv);

  float expected = static_cast<float>(nDev * (nDev + 1) / 2);
  for (int i = 0; i < nDev; i++) {
    std::vector<float> h(recvcount);
    cudaSetDevice(i);
    cudaMemcpy(h.data(), recv[i], recvcount * sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_FLOAT_EQ(h[0], expected) << "rank " << i;
    EXPECT_FLOAT_EQ(h[recvcount - 1], expected) << "rank " << i;
  }
}

// NCCL ReduceScatter with CUDA Graph Stream Capture

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

TEST(NCCL, ReduceScatterGraphCapture) {
  int nDev = 0;
  cudaGetDeviceCount(&nDev);
  if (nDev < 2) GTEST_SKIP() << "Need >= 2 GPUs";

  constexpr int recvcount = 512;
  const int sendcount = recvcount * nDev;
  ReduceScatter rs(nDev);
  Buffer send(nDev, sendcount), recv(nDev, recvcount);
  for (int i = 0; i < nDev; i++) send.fill(i, static_cast<float>(i + 1));

  // Warmup
  rs(send, recv);

  // Capture
  std::vector<std::unique_ptr<CUDAGraph>> graphs(nDev);
  for (int i = 0; i < nDev; i++) {
    cudaSetDevice(i);
    graphs[i] = std::make_unique<CUDAGraph>(rs.streams[i], [&, i] {
      NCCL_CHECK(ncclReduceScatter(send[i], recv[i], recvcount, ncclFloat, ncclSum, rs.comms[i], rs.streams[i]));
    });
  }

  // Reset and replay
  recv.reset();
  CUDAGraph::Launch(graphs, rs.streams, nDev);
  CUDAGraph::Sync(rs.streams, nDev);

  float expected = static_cast<float>(nDev * (nDev + 1) / 2);
  for (int i = 0; i < nDev; i++) {
    std::vector<float> h(recvcount);
    cudaSetDevice(i);
    cudaMemcpy(h.data(), recv[i], recvcount * sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_FLOAT_EQ(h[0], expected) << "rank " << i;
    EXPECT_FLOAT_EQ(h[recvcount - 1], expected) << "rank " << i;
  }
}
