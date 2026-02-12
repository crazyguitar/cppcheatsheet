==================
CUDA Quantization
==================

.. meta::
   :description: CUDA FP8 quantization kernels for LLM inference, covering static, dynamic, and per-token scaling strategies as used in vLLM.
   :keywords: CUDA, FP8, quantization, E4M3, vLLM, inference, LLM, per-token, dynamic scaling, W8A8

.. contents:: Table of Contents
    :backlinks: none

Quantization reduces the precision of tensor values (e.g., FP32/FP16 → FP8) to
cut memory bandwidth and storage by 2–4×, with minimal accuracy loss. Modern LLM
inference engines like vLLM use FP8 quantization on Hopper (H100) and Ada
Lovelace (RTX 4090) GPUs, which have native FP8 tensor core support.

Glossary
--------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Term
     - Definition
   * - **MMA**
     - Matrix Multiply-Accumulate. The fundamental tensor core instruction that
       computes ``D = A × B + C`` on small matrix tiles in hardware. Evolved
       across generations: ``wmma`` (Volta, warp-level) → ``mma`` (Ampere, finer
       PTX control) → ``wgmma`` (Hopper, warp-group level with TMA).
   * - **wgmma**
     - Warp Group MMA. Hopper's tensor core instruction where 4 warps (128
       threads) cooperate on larger tiles, fed asynchronously via TMA.
   * - **TMA**
     - Tensor Memory Accelerator. Hopper hardware unit for async bulk data
       movement between global memory and shared memory, bypassing the register
       file.
   * - **W8A8**
     - Weight 8-bit, Activation 8-bit. Both weights and activations quantized to
       8-bit (FP8 or INT8). The scheme used in vLLM's FP8 path.
   * - **WnA16**
     - Weight n-bit, Activation 16-bit (e.g., W4A16). Weights quantized to n
       bits, activations stay in FP16/BF16. Weight-only quantization.
   * - **AWQ**
     - Activation-aware Weight Quantization. A 4-bit weight-only method that
       selects which channels to keep at higher precision based on activation
       importance rather than weight magnitude.
   * - **GPTQ**
     - Post-training quantization using approximate second-order (Hessian)
       information to minimize quantization error layer by layer. Typically
       4-bit weights.
   * - **Marlin**
     - High-performance 4-bit weight-only GEMM kernel (W4A16) optimized for
       NVIDIA GPUs, used in vLLM for W4A16 inference. See
       `IST-DASLab/marlin <https://github.com/IST-DASLab/marlin>`_.
   * - **SmoothQuant**
     - Technique from `Xiao et al., 2023 <https://arxiv.org/abs/2211.10438>`_
       that migrates quantization difficulty from activations to weights.
       Activations often have outlier channels (a few columns with very large
       values) that make them hard to quantize, while weights are usually smooth.
       SmoothQuant divides activations by a per-channel factor ``s`` and
       multiplies weights by the same ``s``: ``Y = (X / s) @ (W * s)``. After
       smoothing, both sides have similar dynamic ranges and can be quantized to
       INT8/FP8 with less error. The factor ``s`` is calibrated offline from a
       small dataset. Used in vLLM and TensorRT-LLM as a preprocessing step
       before W8A8 INT8 quantization. Less critical for FP8 which handles
       outliers better than INT8.
   * - **mm**
     - Matrix Multiply (GEMM). A single matmul kernel.
   * - **group_mm**
     - Grouped matrix multiply. Batched GEMMs where each batch element can have
       different sizes, used in Mixture-of-Experts (MoE) models.
   * - **c2x / c3x**
     - CUTLASS 2.x / 3.x kernel architecture. c2x uses classic templates; c3x
       uses CuTe layout algebra and wgmma (Hopper). c3x is significantly faster
       on sm_90+.
   * - **E4M3 / E5M2**
     - Two FP8 variants. E4M3 (4 exponent, 3 mantissa, range ±448) for forward
       pass inference. E5M2 (5 exponent, 2 mantissa, range ±57344) for gradients
       during training (wider range, less precision).

FP8 E4M3 Format
----------------

FP8 E4M3 (``__nv_fp8_e4m3``) uses 1 sign + 4 exponent + 3 mantissa bits:

.. code-block:: text

    Bit layout:  S EEEE MMM
    Range:       [-448, 448]
    Precision:   ~3 significant decimal digits near 1.0
    Special:     No Inf (overflow saturates to ±448), NaN exists

Compared to FP16 (range ±65504, 10-bit mantissa), FP8 trades precision for 2×
memory savings. The key insight is that neural network weights and activations
are approximately normally distributed, so most values cluster near zero where
FP8 has adequate precision.

Quantization Math
-----------------

.. code-block:: text

    Quantize:    fp8_val = clamp(fp32_val / scale, -448, 448)
    Dequantize:  fp32_val = fp8_val × scale

The ``scale`` factor maps the tensor's dynamic range into FP8's representable
range. Choosing the right scale is the core challenge:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Strategy
     - Scale Computation
     - Use Case
   * - Static per-tensor
     - ``scale = max(|tensor|) / 448`` (offline calibration)
     - Weights (fixed after training)
   * - Dynamic per-tensor
     - Same formula, computed at runtime
     - Activations (when all tokens share similar range)
   * - Dynamic per-token
     - ``scale[i] = max(|row_i|) / 448`` (one per row)
     - Activations (vLLM W8A8 default, best accuracy)

Static Per-Tensor Quantization
------------------------------

:Source: `src/cuda/quant-fp8-static <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/quant-fp8-static>`_

Static quantization uses a scale factor that is pre-computed offline during a
calibration step and then baked into the model as a constant. This is the
standard approach for **weight quantization** in LLM inference: the model weights
are fixed after training, so their dynamic range is known ahead of time. During
calibration, a representative dataset is run through the model to record the
maximum absolute value of each weight tensor, and the scale is set to
``max(|weights|) / 448``.

At inference time, the kernel simply multiplies each element by the reciprocal of
the scale (``1.0f / scale``), clamps the result to the FP8 representable range
``[-448, 448]``, and converts to ``__nv_fp8_e4m3``. Using the reciprocal avoids a
per-element floating-point division, which is significantly more expensive than
multiplication on GPU hardware.

.. code-block:: cuda

    // Core quantization operation (per-element)
    float inv_scale = 1.0f / (*scale);
    float val = input[i] * inv_scale;
    val = fmaxf(-FP8_E4M3_MAX, fminf(val, FP8_E4M3_MAX));
    out[i] = __nv_fp8_e4m3(val);

The clamping step is critical: FP8 E4M3 has no representation for infinity, so
values outside ``[-448, 448]`` would produce NaN without clamping. The
``__nv_fp8_e4m3()`` constructor performs the actual bit conversion using hardware
instructions on sm_89+ GPUs.

This maps directly to vLLM's ``scaled_fp8_quant_kernel_strided``, which uses the
same multiply-clamp-convert pattern. Dequantization is the reverse: multiply the
FP8 value by the scale to recover an approximate FP32 value. The round-trip error
is bounded by the quantization step size, which depends on the magnitude of the
original value and the scale factor.

Dynamic Per-Tensor Quantization
-------------------------------

:Source: `src/cuda/quant-fp8-dynamic <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/quant-fp8-dynamic>`_

When quantizing **activations** (the intermediate tensors flowing between layers),
the value range is not known ahead of time — it changes with every input. Dynamic
quantization solves this by computing the scale at runtime from the actual tensor
data. This requires a two-pass approach because the scale depends on a global
property (the maximum absolute value) that no single thread can compute alone.

**Pass 1 — Absmax Reduction:** Each thread block scans a portion of the tensor,
finding the local maximum absolute value via a strided loop. The per-thread
maximums are reduced within the block using shared memory. The block's result is
then merged into a global scale value using an atomic operation. Since CUDA has no
native ``atomicMax`` for floats, the standard trick is to reinterpret the float
bits as an integer. This works because IEEE 754 positive floats have the same
ordering as unsigned integers when viewed as bit patterns:

.. code-block:: cuda

    // atomicMax for non-negative floats via int reinterpretation
    // Works because IEEE 754 floats sort the same as ints when positive
    __device__ float atomicMaxFloat(float* addr, float value) {
        return __int_as_float(
            atomicMax((int*)addr, __float_as_int(value)));
    }

The reduction kernel writes ``scale = absmax / FP8_E4M3_MAX`` to device memory.
The scale must be initialized to zero before the kernel launch (via
``cudaMemsetAsync``) so that the atomic max starts from a clean baseline.

**Pass 2 — Quantize:** A second kernel reads the computed scale from device
memory and quantizes every element using the same multiply-clamp-convert pattern
as static quantization. The two-kernel design avoids the need for a global
barrier (which CUDA does not support within a single kernel launch across
blocks). The scale lives in device memory between the two passes, so no
device-to-host round-trip is needed.

.. code-block:: text

    Pass 1: absmax reduction  →  scale = absmax / 448
    Pass 2: quantize using the computed scale

This two-pass pattern matches vLLM's ``segmented_max_reduction_strided`` followed
by ``scaled_fp8_quant_kernel_strided_dynamic``. The overhead of the extra kernel
launch is negligible compared to the memory bandwidth cost of scanning the tensor
twice, but it is the simplest correct approach without cooperative groups or
multi-block synchronization.

Dynamic Per-Token Quantization
------------------------------

:Source: `src/cuda/quant-fp8-per-token <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/quant-fp8-per-token>`_

Per-token quantization assigns a **separate scale to each row** of a
``[tokens × hidden]`` matrix. This is the default activation quantization
strategy in vLLM's W8A8 path
(``dynamic_per_token_scaled_fp8_quant_kernel_strided``) and provides the best
accuracy among the three approaches.

**Why per-token matters:** In a transformer, different tokens in a batch can have
wildly different activation magnitudes. A prompt token early in the sequence
might have activations in the range ``[-5, 5]``, while a later token might spike
to ``[-300, 300]``. With a single per-tensor scale, the scale is dominated by the
largest token, and smaller tokens use only a tiny fraction of the FP8 range,
wasting most of the 3-bit mantissa precision:

.. code-block:: text

    Token 0 (small activations):  scale[0] = 0.02  → full FP8 range utilized
    Token 1 (large activations):  scale[1] = 0.71  → full FP8 range utilized

    vs. per-tensor scale = 0.71 → Token 0 uses only 3% of FP8 range (wasted precision)

**Single-pass kernel design:** Unlike the per-tensor dynamic approach which needs
two separate kernels, per-token quantization can be done in a single kernel
because each block handles exactly one row. There is no cross-block dependency —
each block independently finds its row's absmax, computes the scale, and
quantizes. The three phases within each block are separated by ``__syncthreads()``
barriers:

.. code-block:: cuda

    // Phase 1: Strided scan to find row absmax
    // Each thread reads elements at stride=blockDim.x, tracking its local max.
    // This coalesces memory accesses across threads in the warp.
    float local_max = 0.0f;
    for (int col = tid; col < hidden_size; col += blockDim.x)
        local_max = fmaxf(local_max, fabsf(row_in[col]));

    // Phase 2: Shared memory reduction → thread 0 writes scale[row]
    // Classic parallel reduction: each iteration halves the active threads,
    // combining pairs of maximums until thread 0 holds the row's global max.
    // ... block reduction in shared memory ...
    if (tid == 0) scales[row] = fmaxf(smem[0] / FP8_E4M3_MAX, MIN_SCALE);
    __syncthreads();

    // Phase 3: Quantize with per-row scale
    // All threads read the shared row_scale and quantize their elements.
    // The same strided access pattern ensures coalesced writes to the output.
    float inv_scale = 1.0f / row_scale;
    for (int col = tid; col < hidden_size; col += blockDim.x) {
        float val = row_in[col] * inv_scale;
        row_out[col] = __nv_fp8_e4m3(fmaxf(-448, fminf(val, 448)));
    }

The ``MIN_SCALE`` floor (typically ``1 / (448 * 512)``) prevents division by zero
for all-zero rows. Without it, a row of zeros would produce ``scale = 0``, and
the subsequent ``1.0f / scale`` would yield infinity, corrupting the output.

The output ``scales`` array has shape ``[num_tokens]`` and must be carried
alongside the quantized tensor for dequantization. During matrix multiplication,
the per-token activation scales are combined with per-channel weight scales to
produce the final dequantization factor.

Best Practices
--------------

- Use per-token scaling for activations — it preserves accuracy across tokens
  with different magnitude distributions
- Use static per-tensor scaling for weights — they don't change at inference time
- Always clamp before converting to FP8 to avoid NaN (E4M3 has no Inf)
- Use ``1.0f / scale`` (reciprocal multiply) instead of per-element division
- Initialize the scale to 0 before dynamic reduction kernels
- Guard against all-zero inputs with a minimum scale floor
- FP8 requires ``sm_89+`` (Ada Lovelace) or ``sm_90+`` (Hopper); check with
  ``cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev)``
