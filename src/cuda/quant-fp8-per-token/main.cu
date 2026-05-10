// FP8 Quantization - Dynamic Per-Token Scale
//
// This example demonstrates per-token (per-row) dynamic FP8 quantization.
// Each row of a [tokens × hidden] matrix gets its own scale factor,
// preserving more precision than a single per-tensor scale.
//
// This is the activation quantization strategy used in vLLM's W8A8 path:
// each token's activation vector is independently scaled to maximize
// FP8 dynamic range utilization.
//
// Single-pass approach:
//   1. Each block handles one row (one token)
//   2. First: strided scan to find row absmax via shared memory reduction
//   3. Thread 0 computes scale = absmax / fp8_max, writes to scale[row]
//   4. __syncthreads(), then all threads quantize their elements
//
// Graph structure:
//   [per-token quant kernel]  (one block per token/row)
//
// Reference: vllm/csrc/quantization/w8a8/fp8/common.cu
//            (dynamic_per_token_scaled_fp8_quant_kernel_strided)
//
// Pitfalls:
// - Block count = num_tokens; hidden_size must fit in thread stride
// - Per-token scales array must be pre-allocated with shape [num_tokens]
// - Rows with all zeros get scale=epsilon to avoid division by zero

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>

constexpr float FP8_E4M3_MAX = 448.0f;
constexpr float MIN_SCALE = 1.0f / (FP8_E4M3_MAX * 512.0f);  // floor

// One block per row: find absmax, compute scale, quantize — all in one kernel
__global__ void per_token_fp8_quant(
    __nv_fp8_e4m3* __restrict__ out,  // [tokens, hidden]
    float* __restrict__ scales,       // [tokens]
    const float* __restrict__ in,     // [tokens, hidden]
    int hidden_size
) {
  __shared__ float smem[256];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  const float* row_in = in + row * hidden_size;
  __nv_fp8_e4m3* row_out = out + row * hidden_size;

  // 1) Find row absmax
  float local_max = 0.0f;
  for (int col = tid; col < hidden_size; col += blockDim.x) {
    local_max = fmaxf(local_max, fabsf(row_in[col]));
  }
  smem[tid] = local_max;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
    __syncthreads();
  }

  // 2) Compute and store per-token scale
  __shared__ float row_scale;
  if (tid == 0) {
    row_scale = fmaxf(smem[0] / FP8_E4M3_MAX, MIN_SCALE);
    scales[row] = row_scale;
  }
  __syncthreads();

  // 3) Quantize
  float inv_scale = 1.0f / row_scale;
  for (int col = tid; col < hidden_size; col += blockDim.x) {
    float val = row_in[col] * inv_scale;
    val = fmaxf(-FP8_E4M3_MAX, fminf(val, FP8_E4M3_MAX));
    row_out[col] = __nv_fp8_e4m3(val);
  }
}

// Dequantize with per-token scales
__global__ void
per_token_fp8_dequant(float* __restrict__ out, const __nv_fp8_e4m3* __restrict__ in, const float* __restrict__ scales, int hidden_size) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  float s = scales[row];

  for (int col = tid; col < hidden_size; col += blockDim.x) {
    out[row * hidden_size + col] = float(in[row * hidden_size + col]) * s;
  }
}

TEST(CUDA, QuantFP8PerToken) {
  constexpr int tokens = 32;
  constexpr int hidden = 512;
  constexpr int n = tokens * hidden;
  constexpr int threads = 256;

  // Each row has different magnitude to show per-token benefit
  float h_in[n];
  for (int t = 0; t < tokens; t++) {
    float mag = (t + 1) * 10.0f;  // row 0: ±10, row 31: ±320
    for (int h = 0; h < hidden; h++) {
      h_in[t * hidden + h] = sinf(h * 0.1f) * mag;
    }
  }

  float *d_in, *d_out, *d_scales;
  __nv_fp8_e4m3* d_fp8;
  cudaMalloc(&d_in, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));
  cudaMalloc(&d_fp8, n * sizeof(__nv_fp8_e4m3));
  cudaMalloc(&d_scales, tokens * sizeof(float));

  cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

  per_token_fp8_quant<<<tokens, threads>>>(d_fp8, d_scales, d_in, hidden);
  per_token_fp8_dequant<<<tokens, threads>>>(d_out, d_fp8, d_scales, hidden);
  cudaDeviceSynchronize();

  float h_out[n], h_scales[tokens];
  cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_scales, d_scales, tokens * sizeof(float), cudaMemcpyDeviceToHost);

  // Each row should have its own scale proportional to its magnitude
  EXPECT_LT(h_scales[0], h_scales[tokens - 1]);

  // Per-token should give small relative error even for small-magnitude rows
  for (int t = 0; t < tokens; t++) {
    for (int h = 0; h < hidden; h++) {
      float absval = fabsf(h_in[t * hidden + h]);
      if (absval > 1.0f) {
        float rel_err = fabsf(h_out[t * hidden + h] - h_in[t * hidden + h]) / absval;
        EXPECT_LT(rel_err, 0.15f) << "Row " << t << " col " << h;
      }
    }
  }

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_fp8);
  cudaFree(d_scales);
}
