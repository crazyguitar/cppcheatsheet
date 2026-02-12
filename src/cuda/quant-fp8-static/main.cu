// FP8 Quantization - Static Per-Tensor Scale
//
// This example demonstrates FP8 E4M3 quantization with a pre-computed
// (static) scale factor, as used in inference engines like vLLM for
// weight quantization. The scale is calibrated offline and baked in.
//
// FP8 E4M3 format (IEEE): 1 sign + 4 exponent + 3 mantissa bits
//   - Range: [-448, 448], precision: ~0.125 at magnitude 1
//   - Used by NVIDIA Hopper (H100) and Ada Lovelace (RTX 4090)
//
// Quantization formula:
//   fp8_val = clamp(fp32_val / scale, -fp8_max, fp8_max)
//
// Dequantization formula:
//   fp32_val = fp8_val * scale
//
// Key concepts:
// - __nv_fp8_e4m3 is the hardware FP8 type (CUDA 11.8+, sm_89+)
// - Scale = max(|tensor|) / fp8_max, computed offline for static quant
// - Clamping prevents overflow into NaN/Inf in the FP8 range
// - One block per row, threads stride across columns
//
// Graph structure:
//   [quant kernel]  →  [dequant kernel]
//
// Reference: vllm/csrc/quantization/w8a8/fp8/common.cu
//            (scaled_fp8_quant_kernel_strided)
//
// Pitfalls:
// - __nv_fp8_e4m3 requires -arch=sm_89 or higher
// - Scale of 0 causes division by zero; guard against empty tensors
// - FP8 has no Inf representation in E4M3; overflow saturates to max

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>

constexpr float FP8_E4M3_MAX = 448.0f;

// Quantize: fp32 → fp8 with pre-computed scale
__global__ void static_fp8_quant(
    __nv_fp8_e4m3* __restrict__ out,
    const float* __restrict__ in,
    const float* __restrict__ scale,  // single value
    int n
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float inv_scale = 1.0f / (*scale);
  float val = in[i] * inv_scale;
  val = fmaxf(-FP8_E4M3_MAX, fminf(val, FP8_E4M3_MAX));
  out[i] = __nv_fp8_e4m3(val);
}

// Dequantize: fp8 → fp32
__global__ void fp8_dequant(float* __restrict__ out, const __nv_fp8_e4m3* __restrict__ in, const float* __restrict__ scale, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  out[i] = float(in[i]) * (*scale);
}

TEST(CUDA, QuantFP8Static) {
  constexpr int n = 1024;
  constexpr int threads = 256;
  constexpr int blocks = (n + threads - 1) / threads;

  // Host data
  float h_in[n];
  float absmax = 0.0f;
  for (int i = 0; i < n; i++) {
    h_in[i] = (i - n / 2) * 0.5f;  // range [-256, 255.5]
    absmax = fmaxf(absmax, fabsf(h_in[i]));
  }
  float h_scale = absmax / FP8_E4M3_MAX;  // offline calibration

  // Device alloc
  float *d_in, *d_out, *d_scale;
  __nv_fp8_e4m3* d_fp8;
  cudaMalloc(&d_in, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));
  cudaMalloc(&d_fp8, n * sizeof(__nv_fp8_e4m3));
  cudaMalloc(&d_scale, sizeof(float));

  cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_scale, &h_scale, sizeof(float), cudaMemcpyHostToDevice);

  // Quantize then dequantize
  static_fp8_quant<<<blocks, threads>>>(d_fp8, d_in, d_scale, n);
  fp8_dequant<<<blocks, threads>>>(d_out, d_fp8, d_scale, n);
  cudaDeviceSynchronize();

  // Verify round-trip error is bounded by quantization step
  float h_out[n];
  cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < n; i++) {
    float absval = fabsf(h_in[i]);
    if (absval > 1.0f) {
      EXPECT_LT(fabsf(h_out[i] - h_in[i]) / absval, 0.15f) << "i=" << i << " in=" << h_in[i] << " out=" << h_out[i];
    }
  }

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_fp8);
  cudaFree(d_scale);
}
