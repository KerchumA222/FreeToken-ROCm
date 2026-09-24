// copied from
// https://github.com/vllm-project/vllm/blob/4492e3a55428e161ca8db381edc28263e5da4c8d/csrc/quantization/gguf/mmvq.cuh
// copied and adapted from https://github.com/ggerganov/llama.cpp/blob/b2899/ggml-cuda/mmvq.cu
// NV vectors per block: each block reads its weight rows once and dots them against up to
// NV activation vectors. The donor launched one grid row per vector, re-reading the whole
// weight matrix for every one (a 2-row speculative verify paid 2x on the LM head).
template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda, int NV>
static __global__ void mul_mat_vec_q(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int ncols,
    const int nrows,
    const int nvecs) {
  const auto row = blockIdx.x * blockDim.y + threadIdx.y;
  const int vec0 = blockIdx.y * NV;

  if (row >= nrows || vec0 >= nvecs) {
    return;
  }
  const int nv = min(NV, nvecs - vec0);

  const int blocks_per_row = ncols / qk;
  const int blocks_per_warp = vdr * WARP_SIZE / qi;
  const int nrows_y = (ncols + 512 - 1) / 512 * 512;

  // partial sum for each thread and vector
  float tmp[NV] = {};

  const block_q_t* x = (const block_q_t*)vx;
  const block_q8_1* y = (const block_q8_1*)vy;

  for (auto i = threadIdx.x / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
    const int ibx = row * blocks_per_row + i;  // x block index
    const int iqs = vdr * (threadIdx.x % (qi / vdr));  // x block quant index when casting the quants to int

#pragma unroll
    for (int v = 0; v < NV; ++v) {
      if (v < nv) {
        const int iby = (vec0 + v) * (nrows_y / QK8_1) + i * (qk / QK8_1);  // y block aligned with ibx
        tmp[v] += vec_dot_q_cuda(&x[ibx], &y[iby], iqs);
      }
    }
  }

  // sum up partial sums and write back result
#pragma unroll
  for (int v = 0; v < NV; ++v) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
      tmp[v] += SGLANG_SHFL_XOR_SYNC(uint32_t(-1), tmp[v], mask);
    }
  }

  if (threadIdx.x == 0) {
#pragma unroll
    for (int v = 0; v < NV; ++v) {
      if (v < nv) {
        dst[(vec0 + v) * nrows + row] = tmp[v];
      }
    }
  }
}


// One vector per block for a single vector or a small matrix (more blocks, more
// parallelism); several per block once the weight read dominates.
template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
static void launch_mmvq(const void* vx, const void* vy, scalar_t* dst, const int ncols,
                        const int nrows, const int nvecs, cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  if (nvecs > 1 && nrows >= 16384) {
    const dim3 block_nums(block_num_y, (nvecs + 3) / 4, 1);
    mul_mat_vec_q<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda, 4>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
  } else {
    const dim3 block_nums(block_num_y, nvecs, 1);
    mul_mat_vec_q<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda, 1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
  }
}

template <typename scalar_t>
static void mul_mat_vec_q4_0_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  launch_mmvq<scalar_t, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_q4_1_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  launch_mmvq<scalar_t, QK4_0, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_q5_0_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  launch_mmvq<scalar_t, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_q5_1_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  launch_mmvq<scalar_t, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_q8_0_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  launch_mmvq<scalar_t, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_q2_K_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  launch_mmvq<scalar_t, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_q3_K_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  launch_mmvq<scalar_t, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_q4_K_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  launch_mmvq<scalar_t, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_q5_K_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  launch_mmvq<scalar_t, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_q6_K_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  launch_mmvq<scalar_t, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_iq2_xxs_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  launch_mmvq<scalar_t, QK_K, QI2_XXS, block_iq2_xxs, 1, vec_dot_iq2_xxs_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_iq2_xs_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  launch_mmvq<scalar_t, QK_K, QI2_XS, block_iq2_xs, 1, vec_dot_iq2_xs_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_iq2_s_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  launch_mmvq<scalar_t, QK_K, QI2_S, block_iq2_s, 1, vec_dot_iq2_s_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_iq3_xxs_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  launch_mmvq<scalar_t, QK_K, QI3_XXS, block_iq3_xxs, 1, vec_dot_iq3_xxs_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_iq1_s_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  launch_mmvq<scalar_t, QK_K, QI1_S, block_iq1_s, 1, vec_dot_iq1_s_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_iq1_m_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  launch_mmvq<scalar_t, QK_K, QI1_M, block_iq1_m, 1, vec_dot_iq1_m_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_iq4_nl_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  launch_mmvq<scalar_t, QK4_NL, QI4_NL, block_iq4_nl, VDR_Q4_0_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_iq4_xs_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  launch_mmvq<scalar_t, QK_K, QI4_XS, block_iq4_xs, 1, vec_dot_iq4_xs_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_iq3_s_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  launch_mmvq<scalar_t, QK_K, QI3_XS, block_iq3_s, 1, vec_dot_iq3_s_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}
