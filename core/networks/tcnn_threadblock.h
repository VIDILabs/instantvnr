#pragma once

#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/networks/fully_fused_mlp.h>
#include <tiny-cuda-nn/cutlass_matmul.h>
#include <tiny-cuda-nn/common_device.h>

#include "tcnn_device_api.h"

#include <mma.h>

TCNN_NAMESPACE_BEGIN

template<int WIDTH, int BLOCK_DIM_Z, int N_ITERS, typename OUT_T, Activation ACTIVATION, bool INFERENCE>
__global__ void
kernel_mlp_fused(const Activation output_activation,
                 const __half* __restrict__ input,
                 const __half* __restrict__ weights,
                 OUT_T* __restrict__ out_intermediate,
                 OUT_T* __restrict__ out,
                 const uint32_t batch_size,
                 const uint32_t in_width,
                 const uint32_t out_width,
                 const uint32_t n_hidden_matmuls,
                 const nvcuda::wmma::layout_t input_layout,
                 const nvcuda::wmma::layout_t output_layout);

void
check_shmem_error(cudaError_t error);

template<int WIDTH, typename T, Activation ACTIVATION, bool INFERENCE>
std::enable_if_t<!std::is_same<__half, T>::value>
mlp_fused_forward(cudaStream_t stream,
                  Activation output_activation,
                  const GPUMatrix<T, RM>& weights,
                  const GPUMatrixDynamic<T>& input,
                  GPUMatrix<T>& output_intermediate,
                  GPUMatrixDynamic<T>* output,
                  const uint32_t n_hidden_layers);

template<int WIDTH, typename T, Activation ACTIVATION, bool INFERENCE>
std::enable_if_t<std::is_same<__half, T>::value>
mlp_fused_forward(cudaStream_t stream,
                  Activation output_activation,
                  const GPUMatrix<T, RM>& weights,
                  const GPUMatrixDynamic<T>& input,
                  GPUMatrix<T>& output_intermediate,
                  GPUMatrixDynamic<T>* output,
                  const uint32_t n_hidden_layers);

TCNN_NAMESPACE_END

/* namespace instant neural volume */
namespace vnr {
namespace tcnn_impl {

template <int WIDTH, int BLOCK_DIM_Z, int N_ITERS, typename OUT_T, bool BACKWARD=false>
__device__ void threadblock_layer(Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, OUT_T* __restrict__ out_intermediate_threadblock_this_layer, const OUT_T* __restrict__ activation_aux = nullptr)
{
  using namespace TCNN_NAMESPACE;

  // act_shmem contains the intermediate activations (shared memory) of the thread block's chunk of the batch.
  //           Can be forward activations or backward activations, depending on caller.
  // weights_this_layer points to the weight matrix of the current layer.
  // out_intermediate_threadblock_this_layer points to the location where intermediate activations produced by the thread block should be written to.
  //                  Can be nullptr if nothing should be written.
  // activation_aux points to additional arguments that the activation function may depend on. Points to the hidden forward activations when computing backward activations.

  constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
  constexpr uint32_t N_BLOCKS = WIDTH / 16;

  using namespace nvcuda;

  // If we're performing the backward pass, weights must be loaded in transposed form, which
  // is achieved by interpreting the memory in row_major instead of col_major order.
  using weights_layout_t = std::conditional_t<BACKWARD, wmma::row_major, wmma::col_major>;

  // Fragments
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> act_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, weights_layout_t> weights_frag[N_BLOCKS];
  wmma::fragment<wmma::accumulator, 16, 16, 16, OUT_T> result_frag[N_ITERS];

  // Indices
  const uint32_t li = threadIdx.x; // index in warp ("lane index")
  const uint32_t wi = threadIdx.y; // index in block ("warp index")

  const uint32_t lane_offset = (8 * li) % WIDTH;
  const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

  const uint32_t weights_col = 16 * wi;

  __syncthreads();

  // Load N_BLOCKS chunks of weights from global memory into registers.
  #pragma unroll
  for (uint32_t i = 0; i < N_BLOCKS; ++i) {
    if (BACKWARD) {
      // If we're performing the backward pass, additional index swizzling is needed to
      // load the weights in transposed form.
      wmma::load_matrix_sync(weights_frag[i], weights_this_layer + 16 * i * WIDTH + weights_col, WIDTH);
    } else {
      wmma::load_matrix_sync(weights_frag[i], weights_this_layer + 16 * i + weights_col * WIDTH, WIDTH);
    }
  }

  #pragma unroll
  for (int l = 0; l < N_ITERS; ++l) {
    wmma::fill_fragment(result_frag[l], 0.0f);

    #pragma unroll
    for (uint32_t i = 0; i < N_BLOCKS; ++i) {
      // Load a chunk of intermediate activations from shared memory and multiply with chunk of weights
      wmma::load_matrix_sync(act_frag, act_shmem + 16 * i + (16 * (threadIdx.z + l * BLOCK_DIM_Z)) * (WIDTH + SKEW), WIDTH + SKEW);
      wmma::mma_sync(result_frag[l], act_frag, weights_frag[i], result_frag[l]);
    }

    // Activation
    if (BACKWARD) {
      // Load the temporary forward matrix for the relu transfer
      wmma::load_matrix_sync(act_frag, activation_aux + weights_col + (threadIdx.z + l * BLOCK_DIM_Z) * 16 * WIDTH, WIDTH);
      warp_activation_backward<__half>(activation, result_frag[l], act_frag, result_frag[l]);
    } else {
      warp_activation<__half>(activation, result_frag[l], result_frag[l]);
    }
  }

  __syncthreads();

  #pragma unroll
  for (int l = 0; l < N_ITERS; ++l) {
    wmma::store_matrix_sync(act_shmem + weights_col + (threadIdx.z + l * BLOCK_DIM_Z) * 16 * (WIDTH + SKEW), result_frag[l], WIDTH + SKEW, wmma::mem_row_major);
  }

  if (out_intermediate_threadblock_this_layer != nullptr) {
    __syncthreads();

    #pragma unroll
    for (int l = 0; l < N_ITERS; ++l) {
      *(int4*)&out_intermediate_threadblock_this_layer[lane_offset + (row + 16 * (threadIdx.z + l * BLOCK_DIM_Z)) * WIDTH] = *(int4*)&act_shmem[lane_offset + (row + 16 * (threadIdx.z + l * BLOCK_DIM_Z)) * (WIDTH + SKEW)];
    }
  }
}

template <int WIDTH, int BLOCK_DIM_Z, int N_ITERS>
__device__ void threadblock_load_input_static(__half* __restrict__ act_shmem, const __half* __restrict__ input_threadblock) 
{
  using namespace TCNN_NAMESPACE;

  // act_shmem will be filled by the thread block's chunk of input_threadblock

  constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;

  // Indices
  const uint32_t li = threadIdx.x; // index in warp ("lane index")
  const uint32_t wi = threadIdx.y; // index in block ("warp index")

  const uint32_t lane_offset = (8 * li) % WIDTH;
  const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

  #pragma unroll
  for (int i = 0; i < N_ITERS; ++i) {
    *(int4*)&act_shmem[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * (WIDTH + SKEW)] = *(int4*)&input_threadblock[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * WIDTH];
  }
}

template <int N_POS_DIMS, int N_FEATURES_PER_LEVEL, int WIDTH, int BLOCK_DIM_Z, int N_ITERS>
__device__ void threadblock_load_input_static_with_encoding(__half* __restrict__ act_shmem, const float* __restrict__ coordinate, DeviceNeuralEncoder<__half, N_POS_DIMS, N_FEATURES_PER_LEVEL> encoder)
{
  using namespace TCNN_NAMESPACE;

  // act_shmem will be filled by the thread block's chunk of input_threadblock

  constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;

  // Indices
  const uint32_t li = threadIdx.x; // index in warp ("lane index")
  const uint32_t wi = threadIdx.y; // index in block ("warp index")

  const uint32_t lane_offset = (8 * li) % WIDTH;
  const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

  // #pragma unroll
  // for (int i = 0; i < N_ITERS; ++i) {
  //   *(int4*)&act_shmem[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * (WIDTH + SKEW)] = *(int4*)&input_threadblock[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * WIDTH];
  // }

  // encoding: originally, fully fused MLP allows each thread to load 8 halfs, so in this mode,
  //           each thread will be handling (8/N_FEATURES_PER_LEVEL) encoding levels.
  static_assert(N_FEATURES_PER_LEVEL <= 8, "the number of features per level should be less than 8");
  float coord[N_POS_DIMS];
  
  #pragma unroll
  for (int i = 0; i < N_ITERS; ++i) {
    const uint32_t k = row + 16 * (threadIdx.z + i * BLOCK_DIM_Z);

    __half* encoded = &act_shmem[lane_offset + k * (WIDTH + SKEW)];
    // We are also initializing encoded output to be zero (but this is probably not necessary).
    *(int4*)encoded = {0};

    #pragma unroll
    for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
      coord[dim] = (float)coordinate[dim + k * N_POS_DIMS]; 
    }

    constexpr int N = 8 / N_FEATURES_PER_LEVEL;
    #pragma unroll
    for (uint32_t l = 0; l < N; ++l) {
      const int level = l + N * (lane_offset / 8);
      __half* output_per_level = encoded + l * N_FEATURES_PER_LEVEL;
      // Because of padding, it is possible that level >= encoder.num_levels. Therefore
      // we want to explicitly check level here.
      if (level < encoder.num_levels)
        encoder.encode_one_level(level, coord, encoded + l * N_FEATURES_PER_LEVEL);
    }
  }

}

template <int WIDTH, int BLOCK_DIM_Z, int N_ITERS, typename OUT_T, typename INPUT_LAYOUT>
__device__ void threadblock_input_layer_forward_dynamic(Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ input_threadblock, const __half* __restrict__ weights_this_layer, OUT_T* __restrict__ out_intermediate_threadblock_this_layer, const uint32_t in_width, const uint32_t batch_size) 
{
  using namespace TCNN_NAMESPACE;

  // act_shmem contains the intermediate activations (shared memory) of the thread block's chunk of the batch
  // input_threadblock points to the thread block's chunk of the input batch in global memory
  // weights_this_layer points to the weight matrix of the current layer
  // out_intermediate_threadblock_this_layer points to the location where intermediate activations produced by the thread block should be written to.
  //                  Can be nullptr if nothing should be written.
  // in_width is the dynamic width of the input layer

  constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
  constexpr uint32_t INPUT_SKEW = 8;
  constexpr uint32_t N_BLOCKS = WIDTH / 16;

  using namespace nvcuda;

  // Fragments
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, INPUT_LAYOUT> act_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> weights_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, OUT_T> result_frag[N_ITERS];

  // Indices
  const uint32_t li = threadIdx.x; // index in warp ("lane index")
  const uint32_t wi = threadIdx.y; // index in block ("warp index")

  const uint32_t lane_offset = (8 * li) % WIDTH;
  const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

  const uint32_t weights_col = 16 * wi;

  __half* __restrict__ weights_shmem = act_shmem + BLOCK_DIM_Z * 16 * (in_width + INPUT_SKEW);

  // Load input weight matrix (fits completely into shared memory)
  // Each thread can load 8 fp16 elements (16 bytes) at once; we have N_BLOCKS*BLOCK_DIM_Z warps
  const uint32_t n_elems_per_load = N_BLOCKS * 32 * BLOCK_DIM_Z * 8;
  const uint32_t thread_elem_idx = (li + wi * 32 + threadIdx.z * N_BLOCKS * 32) * 8;

  const uint32_t n_elems_b = WIDTH * in_width;

  #pragma unroll
  for (uint32_t idx = thread_elem_idx; idx < n_elems_b; idx += n_elems_per_load) {
    const uint32_t idx_skewed = idx + idx / in_width * INPUT_SKEW;
    *(int4*)&weights_shmem[idx_skewed] = *(int4*)&weights_this_layer[idx];
  }

  const uint32_t n_tensor_ops = in_width / 16;

  if (std::is_same<INPUT_LAYOUT, wmma::col_major>::value) {
    __syncthreads();
  }

  #pragma unroll
  for (int l = 0; l < N_ITERS; ++l) {
    if (std::is_same<INPUT_LAYOUT, wmma::row_major>::value) {
      // Load chunk of inputs into shmem.
      // This is faster than loading it from gmem directly, even though it is only used once.
      // (Possibly due to latency hiding through staging.)
      const uint32_t n_elems_a = BLOCK_DIM_Z * 16 * in_width;

      #pragma unroll
      for (uint32_t idx = thread_elem_idx; idx < n_elems_a; idx += n_elems_per_load) {
        const uint32_t idx_skewed = idx + idx / in_width * INPUT_SKEW;
        *(int4*)&act_shmem[idx_skewed] = *(int4*)&input_threadblock[l * n_elems_a + idx];
      }

      __syncthreads();
    }

    wmma::fill_fragment(result_frag[l], 0.0f);
    #pragma unroll
    for (uint32_t i = 0; i < n_tensor_ops; ++i) {
      // Load chunk of inputs and weights from shared memory and multiply them
      if (std::is_same<INPUT_LAYOUT, wmma::row_major>::value) {
        wmma::load_matrix_sync(act_frag, act_shmem + 16 * i + (16 * threadIdx.z) * (in_width + INPUT_SKEW), in_width + INPUT_SKEW);
      } else {
        wmma::load_matrix_sync(act_frag, input_threadblock + 16 * i * batch_size + 16 * (threadIdx.z + l * BLOCK_DIM_Z), batch_size);
      }
      wmma::load_matrix_sync(weights_frag, weights_shmem + 16 * i + weights_col * (in_width + INPUT_SKEW), in_width + INPUT_SKEW);
      wmma::mma_sync(result_frag[l], act_frag, weights_frag, result_frag[l]);
    }

    if (std::is_same<INPUT_LAYOUT, wmma::row_major>::value) {
      __syncthreads();
    }

    warp_activation<__half>(activation, result_frag[l], result_frag[l]);
  }

  if (std::is_same<INPUT_LAYOUT, wmma::col_major>::value) {
    __syncthreads();
  }

  #pragma unroll
  for (int l = 0; l < N_ITERS; ++l) {
    wmma::store_matrix_sync(act_shmem + weights_col + (16 * (threadIdx.z + l * BLOCK_DIM_Z)) * (WIDTH + SKEW), result_frag[l], WIDTH + SKEW, wmma::mem_row_major);
  }

  if (out_intermediate_threadblock_this_layer != nullptr) {
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < N_ITERS; ++i) {
      *(int4*)&out_intermediate_threadblock_this_layer[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * WIDTH] = *(int4*)&act_shmem[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * (WIDTH + SKEW)];
    }
  }
}

template <int N_POS_DIMS, int N_FEATURES_PER_LEVEL, int WIDTH, int BLOCK_DIM_Z, int N_ITERS, typename OUT_T, typename INPUT_LAYOUT>
__device__ void threadblock_input_layer_forward_dynamic_with_encoding(Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, const uint32_t in_width, const uint32_t batch_size, const float* __restrict__ coordinate, DeviceNeuralEncoder<__half, N_POS_DIMS, N_FEATURES_PER_LEVEL> encoder) 
{
  using namespace TCNN_NAMESPACE;

  // act_shmem contains the intermediate activations (shared memory) of the thread block's chunk of the batch
  // input_threadblock points to the thread block's chunk of the input batch in global memory
  // weights_this_layer points to the weight matrix of the current layer
  // out_intermediate_threadblock_this_layer points to the location where intermediate activations produced by the thread block should be written to.
  //                  Can be nullptr if nothing should be written.
  // in_width is the dynamic width of the input layer

  constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
  constexpr uint32_t INPUT_SKEW = 8;
  constexpr uint32_t N_BLOCKS = WIDTH / 16;

  using namespace nvcuda;

  // Fragments
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, INPUT_LAYOUT> act_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> weights_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, OUT_T> result_frag[N_ITERS];

  // Indices
  const uint32_t li = threadIdx.x; // index in warp ("lane index")
  const uint32_t wi = threadIdx.y; // index in block ("warp index")

  const uint32_t lane_offset = (8 * li) % WIDTH;
  const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

  const uint32_t weights_col = 16 * wi;

  __half* __restrict__ weights_shmem = act_shmem + BLOCK_DIM_Z * 16 * (in_width + INPUT_SKEW);

  // Load input weight matrix (fits completely into shared memory)
  // Each thread can load 8 fp16 elements (16 bytes) at once; we have N_BLOCKS*BLOCK_DIM_Z warps
  const uint32_t n_elems_per_load = N_BLOCKS * 32 * BLOCK_DIM_Z * 8;
  const uint32_t thread_elem_idx = (li + wi * 32 + threadIdx.z * N_BLOCKS * 32) * 8;

  const uint32_t n_elems_b = WIDTH * in_width;

  #pragma unroll
  for (uint32_t idx = thread_elem_idx; idx < n_elems_b; idx += n_elems_per_load) {
    const uint32_t idx_skewed = idx + idx / in_width * INPUT_SKEW;
    *(int4*)&weights_shmem[idx_skewed] = *(int4*)&weights_this_layer[idx];
  }

  const uint32_t n_tensor_ops = in_width / 16;

  static_assert(std::is_same<INPUT_LAYOUT, wmma::row_major>::value, "only support column major input");

  #pragma unroll
  for (int l = 0; l < N_ITERS; ++l) {
    // Load chunk of inputs into shmem.
    // This is faster than loading it from gmem directly, even though it is only used once.
    // (Possibly due to latency hiding through staging.)
    const uint32_t n_elems_a = BLOCK_DIM_Z * 16 * in_width;

    // encoding: originally, fully fused MLP allows each thread to load 8 halfs, so in this mode,
    //           each thread will be handling (8/N_FEATURES_PER_LEVEL) encoding levels.
    static_assert(N_FEATURES_PER_LEVEL <= 8, "the number of features per level should be less than 8");
    float coord[N_POS_DIMS];

    #pragma unroll
    for (uint32_t idx = thread_elem_idx; idx < n_elems_a; idx += n_elems_per_load) {
      const uint32_t idx_skewed = idx + idx / in_width * INPUT_SKEW;
      __half* encoded = &act_shmem[idx_skewed];
      // We are initializing encoded output to be zero (but this is probably not necessary).
      *(int4*)encoded = {0};

      const uint32_t e = l * n_elems_a + idx;
      const uint32_t in_elem_index  = e / in_width;
      const uint32_t in_lane_offset = e % in_width;

      #pragma unroll
      for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
        coord[dim] = (float)coordinate[dim + in_elem_index * N_POS_DIMS]; 
      }

      constexpr int N = 8 / N_FEATURES_PER_LEVEL;
      #pragma unroll
      for (uint32_t ll = 0; ll < N; ++ll) {
        const int level = ll + N * (in_lane_offset / 8);
        __half* output_per_level = encoded + ll * N_FEATURES_PER_LEVEL;
        // Because of padding, it is possible that level >= encoder.num_levels. Therefore
        // we want to explicitly check level here.
        if (level < encoder.num_levels)
          encoder.encode_one_level(level, coord, output_per_level);
      }

    }

    __syncthreads();

    wmma::fill_fragment(result_frag[l], 0.0f);
    #pragma unroll
    for (uint32_t i = 0; i < n_tensor_ops; ++i) {
      // Load chunk of inputs and weights from shared memory and multiply them
      wmma::load_matrix_sync(act_frag, act_shmem + 16 * i + (16 * threadIdx.z) * (in_width + INPUT_SKEW), in_width + INPUT_SKEW);
      wmma::load_matrix_sync(weights_frag, weights_shmem + 16 * i + weights_col * (in_width + INPUT_SKEW), in_width + INPUT_SKEW);
      wmma::mma_sync(result_frag[l], act_frag, weights_frag, result_frag[l]);
    }

    if (std::is_same<INPUT_LAYOUT, wmma::row_major>::value) {
      __syncthreads();
    }

    warp_activation<__half>(activation, result_frag[l], result_frag[l]);
  }

  #pragma unroll
  for (int l = 0; l < N_ITERS; ++l) {
    wmma::store_matrix_sync(act_shmem + weights_col + (16 * (threadIdx.z + l * BLOCK_DIM_Z)) * (WIDTH + SKEW), result_frag[l], WIDTH + SKEW, wmma::mem_row_major);
  }
}

template <int WIDTH, int BLOCK_DIM_Z, int N_ITERS, typename OUT_T>
__device__ void threadblock_last_layer_forward(Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, OUT_T* __restrict__ out, const uint32_t batch_size, const nvcuda::wmma::layout_t output_layout)
{
  using namespace TCNN_NAMESPACE;
  // act_shmem contains the intermediate activations (shared memory) of the thread block's chunk of the batch
  // weights_this_layer points to the weight matrix of the current layer
  // out points to the location where the result produced by the thread block should be written to.
  //   Can be nullptr if nothing should be written.

  constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
  constexpr uint32_t N_BLOCKS = WIDTH / 16;

  using namespace nvcuda;

  // Fragments
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> act_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> weights_frag[N_BLOCKS];
  wmma::fragment<wmma::accumulator, 16, 16, 16, OUT_T> result_frag;

  // Indices
  const uint32_t li = threadIdx.x; // index in warp ("lane index")
  const uint32_t wi = threadIdx.y; // index in block ("warp index")

  __half* __restrict__ weights_shmem = act_shmem + N_ITERS * BLOCK_DIM_Z * 16 * (WIDTH + SKEW);

  const uint32_t weights_row = (8 * li) % WIDTH;
  const uint32_t weights_col = (8 * li + 8 * 32 * wi) / WIDTH;

  // Load weight matrix into shared memory for the last multiplication.
  // Loading into shared memory as opposed to directly into registers is faster
  // because unlike in the previous layers, each warp uses the same entries of the weight matrix.
  if (threadIdx.z == 0) {
    *(int4*)&weights_shmem[weights_row + weights_col * (WIDTH + SKEW)] = *(int4*)&weights_this_layer[weights_row + weights_col * WIDTH];
  }

  __syncthreads();

  #pragma unroll
  for (uint32_t i = 0; i < N_BLOCKS; ++i)
    wmma::load_matrix_sync(weights_frag[i], weights_shmem + 16 * i, WIDTH + SKEW);

  // Perform last layer by parallelizing over iters
  for (uint32_t idx = wi; idx < N_ITERS; idx += N_BLOCKS) {
    wmma::fill_fragment(result_frag, 0.0f);
    #pragma unroll
    for (uint32_t i = 0; i < N_BLOCKS; ++i) {
      // Load a chunk of intermediate activations from shared memory and multiply with chunk of the weight matrix
      wmma::load_matrix_sync(act_frag, act_shmem + 16 * i + (16 * (threadIdx.z + idx * BLOCK_DIM_Z)) * (WIDTH + SKEW), WIDTH + SKEW);
      wmma::mma_sync(result_frag, act_frag, weights_frag[i], result_frag);
    }

    warp_activation<__half>(activation, result_frag, result_frag);

    if (output_layout == wmma::mem_row_major) {
      wmma::store_matrix_sync(out + (threadIdx.z + idx * BLOCK_DIM_Z) * 16 * 16, result_frag, 16, output_layout);
    } else {
      wmma::store_matrix_sync(out + (threadIdx.z + idx * BLOCK_DIM_Z) * 16, result_frag, batch_size, output_layout);
    }
  }
}

// template <int WIDTH, int BLOCK_DIM_Z, int N_ITERS>
// __device__ void threadblock_write_output_static(const __half* __restrict__ act_shmem, __half* __restrict__ output_threadblock) 
// {
//   // output_threadblock will be filled by the thread block's act_shmem
// 
//   constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
// 
//   // Indices
//   const uint32_t li = threadIdx.x; // index in warp ("lane index")
//   const uint32_t wi = threadIdx.y; // index in block ("warp index")
// 
//   const uint32_t lane_offset = (8 * li) % WIDTH;
//   const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;
// 
//   __syncthreads();
// 
//   #pragma unroll
//   for (int i = 0; i < N_ITERS; ++i) {
//     *(int4*)&output_threadblock[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * WIDTH] = *(int4*)&act_shmem[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * (WIDTH + SKEW)];
//   }
// }

template <int N_POS_DIMS, int WIDTH, int BLOCK_DIM_Z, int N_ITERS>
__device__ void threadblock_read_pos_from_shmem(float* __restrict__ pos_shmem, const float* __restrict__ pos_threadblock, const uint32_t in_width) 
{
  constexpr uint32_t N_BLOCKS = WIDTH / 16;

  const uint32_t li = threadIdx.x; // index in warp ("lane index")
  const uint32_t wi = threadIdx.y; // index in block ("warp index")

  const uint32_t n_elems_per_load = 32 * N_BLOCKS * BLOCK_DIM_Z * 8 / in_width;
  const uint32_t thread_elem_idx = (li + wi * 32 + threadIdx.z * N_BLOCKS * 32) * 8  / in_width;

  for (int l = 0; l < N_ITERS; ++l) {
    const uint32_t n_elems_a = BLOCK_DIM_Z * 16;
  
    #pragma unroll
    for (uint32_t idx = thread_elem_idx; idx < n_elems_a; idx += n_elems_per_load) {
      const uint32_t e = l * n_elems_a + idx;
  
      #pragma unroll
      for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
        pos_shmem[dim + e * N_POS_DIMS] = (float)pos_threadblock[dim + e * N_POS_DIMS]; 
      }
    }
  }
}

template <int N_POS_DIMS, int WIDTH, int BLOCK_DIM_Z, int N_ITERS>
__device__ void threadblock_read_pos_from_shmem(float* __restrict__ pos_shmem, const float3& pos) 
{
  constexpr uint32_t N_BLOCKS = WIDTH / 16;
  static_assert(N_POS_DIMS == 3, "N_POS_DIMS == 3");
  static_assert(N_ITERS == 2 * N_BLOCKS, "N_ITERS == 2 * N_BLOCKS");

  const uint32_t i = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;

  pos_shmem[0 + i * 3] = pos.x;
  pos_shmem[1 + i * 3] = pos.y;
  pos_shmem[2 + i * 3] = pos.z;

  __syncthreads();
}

template <int WIDTH, int BLOCK_DIM_Z, int N_ITERS, typename OUT_T>
__device__ void threadblock_write_output_from_shmem(const OUT_T* __restrict__ out_shmem, OUT_T* __restrict__ output_threadblock, const uint32_t batch_size, const nvcuda::wmma::layout_t output_layout) 
{
  constexpr uint32_t N_BLOCKS = WIDTH / 16;

  const uint32_t lane_linear_index = 8 * threadIdx.x;
  const uint32_t lane_offset       = lane_linear_index % 16;
  const uint32_t row               = lane_linear_index / 16;
  
  for (uint32_t idx = threadIdx.y; idx < N_ITERS; idx += N_BLOCKS) {
    if (output_layout == nvcuda::wmma::mem_row_major) {
      const uint32_t k = row + 16 * (threadIdx.z + idx * BLOCK_DIM_Z);
      *(int4*)&output_threadblock[lane_offset + k * 16] = *(int4*)&out_shmem[lane_offset + k * 16];
      // wmma::store_matrix_sync(out + (threadIdx.z + idx * BLOCK_DIM_Z) * 16 * 16, result_frag, 16, output_layout);
    } else {
      const uint32_t k = 16 * (threadIdx.z + idx * BLOCK_DIM_Z);
      *(int4*)&output_threadblock[k + batch_size * row + lane_offset] = *(int4*)&out_shmem[k + (16 * BLOCK_DIM_Z * N_ITERS) * row + lane_offset];
      // wmma::store_matrix_sync(out + (threadIdx.z + idx * BLOCK_DIM_Z) * 16, result_frag, batch_size, output_layout);
    }
  }
}

template <int WIDTH, int BLOCK_DIM_Z, int N_ITERS, typename OUT_T>
__device__ void threadblock_write_output_from_shmem(const OUT_T* __restrict__ out_shmem, OUT_T* __restrict__ out) 
{
  constexpr uint32_t N_BLOCKS = WIDTH / 16;
  static_assert(N_ITERS == 2 * N_BLOCKS, "N_ITERS == 2 * N_BLOCKS");

  __syncthreads();

  const uint32_t i = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;

  struct Type16Half { int4 a, b; };
  static_assert(sizeof(Type16Half) == 16 * sizeof(__half), "we are copying 16 halfs");
  *(Type16Half*)&out[0] = *(Type16Half*)&out_shmem[i * 16];

  // *(int4*)&out[0] = *(int4*)&out_shmem[i * 16 + 0];
  // *(int4*)&out[8] = *(int4*)&out_shmem[i * 16 + 8];
}

}
}
