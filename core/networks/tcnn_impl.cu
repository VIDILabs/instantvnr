//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //

#ifdef ENABLE_IN_SHADER
#include <tiny-cuda-nn/cutlass_matmul.h>
#include <mma.h>
#include "tcnn_threadblock.h"
#include "tcnn_device_api.h"
#endif

#include "tcnn_network.h"

#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/networks/fully_fused_mlp.h>
#include <tiny-cuda-nn/common_device.h>

/* namespace instant neural volume */
namespace vnr {
namespace tcnn_impl {

#ifdef ENABLE_IN_SHADER

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

template<typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t WIDTH>
__device__ T
DeviceNeuralVolume<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, WIDTH>::sample(const float3 coordinate) const
{
  // clang-format off

  typedef T OUT_T;

  const DeviceNeuralEncoder<__half, N_POS_DIMS, N_FEATURES_PER_LEVEL>& enc = m_encoder;
  const DeviceNeuralNetwork<__half, WIDTH>& mlp = m_network;
  const uint32_t in_width = m_n_internal_features;

  constexpr uint32_t N_BLOCKS = WIDTH / 16;
  constexpr uint32_t shmem_size_coord  = sizeof(float) * (16 * BLOCK_DIM_Z * N_ITERS) * N_POS_DIMS;
  constexpr uint32_t shmem_size_output = sizeof(OUT_T) * (16 * BLOCK_DIM_Z * N_ITERS) * 16;

  static_assert(N_POS_DIMS == 3, "N_POS_DIMS must be 3 for volumes");
  static_assert(2 * N_BLOCKS == N_ITERS, "this has to be true: 2 * N_BLOCKS == N_ITERS");

  // Shared memory contains the intermediate activations of blockDim.y*16 elements.
  // In some cases, it also contains the weight matrix for the first and last layer.
  extern __shared__ __half shmem[];

  float*  pos_shmem = (float*)shmem;
  OUT_T*  out_shmem = shmem + shmem_size_coord / sizeof(__half);
  __half* act_shmem = shmem + (shmem_size_coord + shmem_size_output) / sizeof(__half);

  // Each block computes exactly one 16-element chunk of the batch.
  const uint32_t elem_idx = blockIdx.x * 16 * BLOCK_DIM_Z * N_ITERS;

  const __half* __restrict__ weights = mlp.weights;
  const auto ACTIVATION = mlp.activation;
  const auto output_activation = mlp.output_activation;
  const auto n_hidden_matmuls  = mlp.n_hidden_matmuls; 

  // assert(__activemask() == FULL_MASK && "all threads should be active");

  OUT_T out[16] = {};

  // -------------------------------------------------------------------------------------------------
  //
  // -------------------------------------------------------------------------------------------------
  // assert(input_layout == nvcuda::wmma::mem_row_major && "only accept column major input");

  threadblock_read_pos_from_shmem<N_POS_DIMS, WIDTH, BLOCK_DIM_Z, N_ITERS>(pos_shmem, (float3&)coordinate);

  // First layer
  if (in_width != WIDTH) {
    threadblock_input_layer_forward_dynamic_with_encoding<N_POS_DIMS, N_FEATURES_PER_LEVEL, WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T, nvcuda::wmma::row_major>(ACTIVATION, act_shmem, weights, in_width, 0, pos_shmem, enc);
  } else {
    threadblock_load_input_static_with_encoding<N_POS_DIMS, N_FEATURES_PER_LEVEL, WIDTH, BLOCK_DIM_Z, N_ITERS>(act_shmem, pos_shmem, enc);
    threadblock_layer<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(ACTIVATION, act_shmem, weights, nullptr);
  }

  // Hidden layers
  const uint32_t first_layer_size = WIDTH * in_width;
  const uint32_t layer_stride = WIDTH * WIDTH;
  for (uint32_t k = 0; k < n_hidden_matmuls; ++k) {
    threadblock_layer<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(ACTIVATION, act_shmem, weights + first_layer_size + layer_stride * k, nullptr);
  }

  threadblock_last_layer_forward<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(output_activation, act_shmem, weights + first_layer_size + layer_stride * n_hidden_matmuls, out_shmem, 16, nvcuda::wmma::mem_row_major);

  threadblock_write_output_from_shmem<WIDTH, BLOCK_DIM_Z, N_ITERS>(out_shmem, out);

  return out[0];

  // clang-format on
}

template<uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t WIDTH, typename OUT_T>
__global__ void
DeviceNeuralVolume_batch_inference_kernel(
  const DeviceNeuralVolume<__half, N_POS_DIMS, N_FEATURES_PER_LEVEL, WIDTH> volume,
  const float* __restrict__ coordinate,
  OUT_T* __restrict__ output,
  const uint32_t batch_size,
  const uint32_t in_width,
  const uint32_t out_width,
  const nvcuda::wmma::layout_t input_layout,
  const nvcuda::wmma::layout_t output_layout)
{
  // clang-format off

  using namespace TCNN_NAMESPACE;

  const DeviceNeuralEncoder<__half, N_POS_DIMS, N_FEATURES_PER_LEVEL>& enc = volume.encoder();
  const DeviceNeuralNetwork<__half, WIDTH>& mlp                            = volume.network();

  constexpr int BLOCK_DIM_Z = volume.n_block_dim_z();
  constexpr int N_ITERS     = volume.n_iters();

  // `input` points to the input matrix. Can be any width.
  // `weights` points to the weight matrices (contiguous in memory).
  // `output_intermediate` points to the memory where intermediate activations should be written. When performing inference, a value of nullptr is expected (intermediate results are not written).
  // `output` points to the memory where the network output should be written. (Output width is assumed to be 16 neurons.)

  // Commented out due to isolated strange side-effects on Windows
  // if (INFERENCE) {
  //   assert(output_intermediate == nullptr);
  // } else {
  //   assert(output_intermediate);
  // }

  // constexpr uint32_t N_BLOCKS = WIDTH / 16;
  constexpr uint32_t shmem_size_coord  = sizeof(float) * (16 * BLOCK_DIM_Z * N_ITERS) * N_POS_DIMS;
  constexpr uint32_t shmem_size_output = sizeof(OUT_T) * (16 * BLOCK_DIM_Z * N_ITERS) * 16;

  // Shared memory contains the intermediate activations of blockDim.y*16 elements.
  // In some cases, it also contains the weight matrix for the first and last layer.
  extern __shared__ __half shmem[];

  float*  pos_shmem = (float*)shmem;
  OUT_T*  out_shmem = shmem + shmem_size_coord / sizeof(__half);
  __half* act_shmem = shmem + (shmem_size_coord + shmem_size_output) / sizeof(__half);

  // Each block computes exactly one 16-element chunk of the batch.
  const uint32_t elem_idx = blockIdx.x * 16 * BLOCK_DIM_Z * N_ITERS;

  const __half* __restrict__ weights = mlp.weights; 
  const Activation ACTIVATION = mlp.activation;
  const Activation output_activation = mlp.output_activation;
  const uint32_t n_hidden_matmuls = mlp.n_hidden_matmuls; 

  // -------------------------------------------------------------------------------------------------
  //
  // -------------------------------------------------------------------------------------------------

#if 1

  // const float* __restrict__ pos = coordinate + elem_idx * N_POS_DIMS;
  // {
  //   const uint32_t li = threadIdx.x; // index in warp ("lane index")
  //   const uint32_t wi = threadIdx.y; // index in block ("warp index")
  //   const uint32_t n_elems_per_load = 32 * N_BLOCKS *BLOCK_DIM_Z * 8 / in_width;
  //   const uint32_t thread_elem_idx = (li + wi * 32 + threadIdx.z * N_BLOCKS * 32) * 8  / in_width;
  //   for (int l = 0; l < N_ITERS; ++l) {
  //     const uint32_t n_elems_a = BLOCK_DIM_Z * 16;
  //     #pragma unroll
  //     for (uint32_t idx = thread_elem_idx; idx < n_elems_a; idx += n_elems_per_load) {
  //       const uint32_t e = l * n_elems_a + idx;
  //       #pragma unroll
  //       for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
  //         pos_shmem[dim + e * N_POS_DIMS] = (float)pos[dim + e * N_POS_DIMS]; 
  //       }
  //     }
  //   }
  // }
  // threadblock_read_pos_from_shmem<N_POS_DIMS, WIDTH, BLOCK_DIM_Z, N_ITERS>(pos_shmem, pos);

  // threadblock_read_pos_from_shmem<N_POS_DIMS, WIDTH, BLOCK_DIM_Z, N_ITERS>(pos_shmem, coordinate + elem_idx * N_POS_DIMS, in_width);

  // -------------------------------------------------------------------------------------------------
  //
  // -------------------------------------------------------------------------------------------------
  assert(input_layout == nvcuda::wmma::mem_row_major && "only accept column major input");

  // First layer
  if (in_width != WIDTH) {
    // CHANGE: only support inference
    // threadblock_input_layer_forward_dynamic<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T, nvcuda::wmma::row_major>(ACTIVATION, act_shmem, input + elem_idx * in_width, weights, !INFERENCE ? (out_intermediate + elem_idx * WIDTH) : nullptr, in_width, batch_size);
    // threadblock_input_layer_forward_dynamic<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T, nvcuda::wmma::row_major>(ACTIVATION, act_shmem, input + elem_idx * in_width, weights, nullptr, in_width, batch_size);
    threadblock_input_layer_forward_dynamic_with_encoding<N_POS_DIMS, N_FEATURES_PER_LEVEL, WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T, nvcuda::wmma::row_major>(ACTIVATION, act_shmem, weights, in_width, batch_size, coordinate + elem_idx * N_POS_DIMS, enc);
    // threadblock_input_layer_forward_dynamic_with_encoding<N_POS_DIMS, N_FEATURES_PER_LEVEL, WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T, nvcuda::wmma::row_major>(ACTIVATION, act_shmem, weights, in_width, batch_size, pos_shmem, enc);
  } else {
    // If the input has the same width & layout as the hidden layers, we can simply use the network's regular layer routine (with static size) instead of using the slower dynamic input layer routine.
    // threadblock_load_input_static<WIDTH, BLOCK_DIM_Z, N_ITERS>(act_shmem, input + elem_idx * WIDTH);
    threadblock_load_input_static_with_encoding<N_POS_DIMS, N_FEATURES_PER_LEVEL, WIDTH, BLOCK_DIM_Z, N_ITERS>(act_shmem, coordinate + elem_idx * N_POS_DIMS, enc);
    // threadblock_load_input_static_with_encoding<N_POS_DIMS, N_FEATURES_PER_LEVEL, WIDTH, BLOCK_DIM_Z, N_ITERS>(act_shmem, pos_shmem, enc);

    // CHANGE: only support inference
    // threadblock_layer<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(ACTIVATION, act_shmem, weights, !INFERENCE ? (output_intermediate + elem_idx * WIDTH) : nullptr);
    threadblock_layer<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(ACTIVATION, act_shmem, weights, nullptr);
  }

  const uint32_t first_layer_size = WIDTH * in_width;
  const uint32_t layer_stride = WIDTH * WIDTH;
  const uint32_t output_stride = WIDTH * batch_size;

  // Hidden layers
  for (uint32_t k = 0; k < n_hidden_matmuls; ++k) {
    // CHANGE: only support inference
    // threadblock_layer<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(ACTIVATION, act_shmem, weights + first_layer_size + layer_stride * k, !INFERENCE ? (output_intermediate + output_stride * (k + 1) + elem_idx * WIDTH) : nullptr);
    threadblock_layer<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(ACTIVATION, act_shmem, weights + first_layer_size + layer_stride * k, nullptr);
  }

  // CHANGE: no need to support that many output layers
  assert(out_width <= 16 && "do not support more than 16 output layers");
  // if (out_width > 16) {
  //   // In the forward pass, intermediate activations are already written out.
  //   // if (INFERENCE) {
  //   //   threadblock_write_output_static<WIDTH, BLOCK_DIM_Z, N_ITERS>(act_shmem, output_intermediate + elem_idx * WIDTH);
  //   // }
  //  // CHANGE: only support inference
  //  threadblock_write_output_static<WIDTH, BLOCK_DIM_Z, N_ITERS>(act_shmem, output_intermediate + elem_idx * WIDTH);
  // } else if (output) {
  //   // Last layer
  //   if (output_layout == nvcuda::wmma::mem_row_major) {
  //     threadblock_last_layer_forward<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(output_activation, act_shmem, weights + first_layer_size + layer_stride * n_hidden_matmuls, output + elem_idx * 16, 16, output_layout);
  //   } else {
  //     threadblock_last_layer_forward<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(output_activation, act_shmem, weights + first_layer_size + layer_stride * n_hidden_matmuls, output + elem_idx, batch_size, output_layout);
  //   }
  // }

  // CHANGE: focus on inference only
  assert(output && "output cannot be NULL for inference");
  if (output_layout == nvcuda::wmma::mem_row_major) {
    threadblock_last_layer_forward<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(output_activation, act_shmem, weights + first_layer_size + layer_stride * n_hidden_matmuls, output + elem_idx * 16, 16, output_layout);
    // threadblock_last_layer_forward<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(output_activation, act_shmem, weights + first_layer_size + layer_stride * n_hidden_matmuls, out_shmem, 16, output_layout);
  } else {
    threadblock_last_layer_forward<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(output_activation, act_shmem, weights + first_layer_size + layer_stride * n_hidden_matmuls, output + elem_idx, batch_size, output_layout);
    // threadblock_last_layer_forward<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(output_activation, act_shmem, weights + first_layer_size + layer_stride * n_hidden_matmuls, out_shmem, 16 * BLOCK_DIM_Z * N_ITERS, output_layout);
  }

  // CHANGE: copy output data from shared memory to global memory
  // OUT_T* __restrict__ out = (output_layout == nvcuda::wmma::mem_row_major) ? output + elem_idx * 16 : output + elem_idx;
  // {
  //   const uint32_t lane_linear_index = 8 * threadIdx.x;
  //   const uint32_t lane_offset       = lane_linear_index % 16;
  //   const uint32_t row               = lane_linear_index / 16;
  // 
  //   for (uint32_t idx = threadIdx.y; idx < N_ITERS; idx += N_BLOCKS) {
  //     if (output_layout == nvcuda::wmma::mem_row_major) {
  //       const uint32_t k = row + 16 * (threadIdx.z + idx * BLOCK_DIM_Z);
  //       *(int4*)&out[lane_offset + k * 16] = *(int4*)&out_shmem[lane_offset + k * 16];
  //       // wmma::store_matrix_sync(out + (threadIdx.z + idx * BLOCK_DIM_Z) * 16 * 16, result_frag, 16, output_layout);
  //     } else {
  //       const uint32_t k = 16 * (threadIdx.z + idx * BLOCK_DIM_Z);
  //       *(int4*)&out[k + batch_size * row + lane_offset] = *(int4*)&out_shmem[k + (16 * BLOCK_DIM_Z * N_ITERS) * row + lane_offset];
  //       // wmma::store_matrix_sync(out + (threadIdx.z + idx * BLOCK_DIM_Z) * 16, result_frag, batch_size, output_layout);
  //     }
  //   }
  // }

  // threadblock_write_output_from_shmem<WIDTH, BLOCK_DIM_Z, N_ITERS>(out_shmem, (output_layout == nvcuda::wmma::mem_row_major) ? output + elem_idx * 16 : output + elem_idx, batch_size, output_layout);
  // threadblock_write_output_from_shmem<WIDTH, BLOCK_DIM_Z, N_ITERS>(out_shmem, &_out);

#else

  const uint32_t i = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
  const float* __restrict__ cc = coordinate + elem_idx * N_POS_DIMS;
  float3 pos;
  pos.x = cc[0 + i * 3];
  pos.y = cc[1 + i * 3];
  pos.z = cc[2 + i * 3];
  __syncthreads();
  OUT_T* __restrict__ out = output + elem_idx * 16;
  out[16 * i] = volume.sample(pos);

#endif

  // clang-format on
}

template<typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t WIDTH>
void DeviceNeuralVolume<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, WIDTH>::batch_sample_internal(
  cudaStream_t stream,
  const Activation ACTIVATION,
  const GPUMatrixDynamic<float>& coord,
  GPUMatrixDynamic<T>& output
) {
  // clang-format off
  static_assert(std::is_same<__half, T>::value, "only support half");

  DeviceNeuralEncoder<T, N_POS_DIMS, N_FEATURES_PER_LEVEL>& enc = m_encoder;
  DeviceNeuralNetwork<T, WIDTH>& mlp                            = m_network;

  using namespace TCNN_NAMESPACE;

  // assert(coord.n() == input.cols() && "coordinate and encoded input buffer must match");
  // if (enc.output_layout == RM) {
  //   linear_kernel(encode_kernel<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, true>, 0, stream, coord.n(), enc, coord.data(), input.data());
  // }
  // else {
  //   linear_kernel(encode_kernel<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, false>, 0, stream, coord.n(), enc, coord.data(), input.data());
  // }

  const uint32_t batch_size = coord.cols();
  const uint32_t in_width   = mlp.n_input_width;
  const uint32_t out_width  = output.rows();

  static_assert(WIDTH < 256, "maximum WIDTH == 128");
  constexpr uint32_t SKEW         = 8; /*WIDTH % 16 == 0 ? 8 : 0;*/ // <- always going to be 8 as we only support multiple-of-16 widths
  constexpr uint32_t INPUT_SKEW   = 8; // <- likewise with inputs
  constexpr uint32_t N_BLOCK_ROWS = WIDTH / 16;
  constexpr uint32_t N_ITERS      = 2 * N_BLOCK_ROWS; // 8; /*WIDTH >= 256 ? 2 : 8;*/ // <- always 8 because maximum WIDTH == 128
  constexpr uint32_t BLOCK_DIM_Z  = 1; // WIDTH == 128 ? 2 : 1;
  static_assert(WIDTH % 16 == 0, "Width must be a multiply of 16.");
  if (in_width % 16 != 0) throw std::runtime_error{"Inputs must have a multiple-of-16 elements."};
  if (output.cols() != batch_size) throw std::runtime_error{"Batch size of inputs and outputs doesn't match."};
  if (batch_size % (16 * N_ITERS * BLOCK_DIM_Z) != 0) throw std::runtime_error{"Batch size must be a multiple of " + std::to_string(16 * N_ITERS * BLOCK_DIM_Z) + "."};

  const dim3 threads = { 32u, N_BLOCK_ROWS, BLOCK_DIM_Z }; // 32 threads = 1 warp, N_BLOCK_ROWS warps per block for 16 rows, up to 2x 8 warps can share input (does not help vs. 1)
  uint32_t n_elems_per_block = 16 * BLOCK_DIM_Z * N_ITERS;
  uint32_t n_blocks = TCNN_NAMESPACE :: div_round_up(batch_size, n_elems_per_block);

  constexpr uint32_t shmem_size_coord  = sizeof(float ) * (16 * BLOCK_DIM_Z * N_ITERS) * N_POS_DIMS;
  constexpr uint32_t shmem_size_output = sizeof(__half) * (16 * BLOCK_DIM_Z * N_ITERS) * 16;

  // 16*WIDTH rows of weights (for the last layer; others are in registers only) + 16*WIDTH*BLOCK_DIM_Z*N_ITERS rows of intermediate activations
  size_t shmem_size = sizeof(__half) * (16 + 16 * BLOCK_DIM_Z * N_ITERS) * (WIDTH + SKEW); 
  // If the input width is dynamic, the input weight matrix as well as part of the input will live in extra shared memory
  if (in_width != WIDTH) {
    shmem_size = std::max(shmem_size, sizeof(__half) * (WIDTH + 16 * BLOCK_DIM_Z) * (in_width + INPUT_SKEW));
  }
  shmem_size += shmem_size_coord + shmem_size_output;

  const dim3 blocks = { n_blocks, 1u, 1u };
  check_shmem_error(cudaFuncSetAttribute(DeviceNeuralVolume_batch_inference_kernel<N_POS_DIMS, N_FEATURES_PER_LEVEL, WIDTH, __half>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_size));
  DeviceNeuralVolume_batch_inference_kernel<N_POS_DIMS, N_FEATURES_PER_LEVEL, WIDTH, __half><<<blocks, threads, shmem_size, stream>>>(
    *this, coord.data(), output.data(), /*enc, mlp,*/ batch_size, in_width, out_width,
    // The kernels operate with transposed layouts compared with the MLP code
    nvcuda::wmma::mem_row_major, output.layout() == RM ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major
  );

  // clang-format on
}

template<typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t WIDTH>
void
DeviceNeuralVolume<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, WIDTH>::batch_sample(cudaStream_t stream,
                                                                             const GPUMatrixDynamic<float>& coord,
                                                                             GPUMatrixDynamic<float>& _output)
{
  // clang-format off
  GridEncodingTemplated<T, N_POS_DIMS, N_FEATURES_PER_LEVEL>* enc = m_encoder.self;
  FullyFusedMLP<T, WIDTH>* mlp                                    = m_network.self;

  using namespace TCNN_NAMESPACE;

  // encoded input for MLP
  // GPUMatrixDynamic<T> input = { enc->num_encoded_dims(), coord.n(), stream, enc->output_layout() };

  const uint32_t input_width = enc->num_encoded_dims();

  /* ---------------------------------------------------------------------------------------------- */
  /* Encoding                                                                                       */
  /* ---------------------------------------------------------------------------------------------- */
  
  // look like the input coordinate layout is assumed to be CM but not enforced in TCNN, so check here.
  assert(coord.layout() == CM && "input coordinate should be a column major matrix");
  assert(coord.m() == N_POS_DIMS && "incorrect coordinate buffer shape");

  const uint32_t n_elements = coord.n();
  if (enc->m_n_padded_output_dims == 0 || n_elements == 0) { return; }

  // PitchedPtr<T> input_pitched{ input.data(), input.m() };
  // if (enc->m_n_to_pad > 0) {
  //   parallel_for_gpu_aos(stream, n_elements, enc->m_n_to_pad,
  //     [n_output_dims = enc->m_n_output_dims, input_pitched] __device__(size_t elem, size_t dim) {
  //       input_pitched(elem)[n_output_dims + dim] = 0;
  //     });
  // }

  // DeviceNeuralEncoder<T, N_POS_DIMS, N_FEATURES_PER_LEVEL> d_enc(enc);

  /* ---------------------------------------------------------------------------------------------- */
  /* MLP                                                                                            */
  /* ---------------------------------------------------------------------------------------------- */

  // CHANGE: rename inference_output_tmp -> outputs
  GPUMatrixDynamic<T> output{ mlp->m_padded_output_width, _output.n(), stream, _output.layout() };

  // BEGIN INLINE

  // Various error checks
  if (input_width   != mlp->m_input_width)      throw std::runtime_error(std::string("Input has incorrect width: ") + std::to_string(input_width) + "!=" + std::to_string(mlp->m_input_width));
  if (output.m() != mlp->m_padded_output_width) throw std::runtime_error(std::string("Output has incorrect width: ") + std::to_string(output.m()) + "!=" + std::to_string(mlp->m_output_width));
  if (coord.n()  != output.n())                 throw std::runtime_error(std::string("Input and output don't have matching batch size: ") + std::to_string(coord.n()) + "!=" + std::to_string(output.n()));

  // Make sure our temporary buffers have the correct size for the given batch size
  // DeviceNeuralNetwork<T, WIDTH> d_mlp(mlp);

  switch (mlp->m_activation) {
    case Activation::None:        batch_sample_internal(stream, Activation::None       , coord, /*input,*/ output); break;
    case Activation::Exponential: batch_sample_internal(stream, Activation::Exponential, coord, /*input,*/ output); break;
    case Activation::Sigmoid:     batch_sample_internal(stream, Activation::Sigmoid    , coord, /*input,*/ output); break;
    case Activation::ReLU:        batch_sample_internal(stream, Activation::ReLU       , coord, /*input,*/ output); break;
    case Activation::Squareplus:  batch_sample_internal(stream, Activation::Squareplus , coord, /*input,*/ output); break;
    case Activation::Softplus:    batch_sample_internal(stream, Activation::Softplus   , coord, /*input,*/ output); break;
    default: throw std::runtime_error{"Unsupported activation."};
  }

  assert(mlp->m_output_width <= 16 && "do not support more than 16 output layers");

  // END INLINE
  
  // convert output from T to float
  const uint32_t n_output_elements = (uint32_t)_output.n_elements();
  // If the layout is row major, trimming away excess dimensions amounts to simply discarding the tail of the buffer.
  if (_output.layout() == RM) {
    // cast_from<T><<<n_blocks_linear(n_output_elements), n_threads_linear, 0, stream>>>(n_output_elements, output.data(), _output.data());
    linear_kernel(cast_from<T>, 0, stream, n_output_elements, output.data(), _output.data());
  }
  else {
    // trim_and_cast<T><<<n_blocks_linear(n_output_elements), n_threads_linear, 0, stream>>>(n_output_elements, mlp->m_padded_output_width, mlp->m_output_width, output.data(), _output.data());
    linear_kernel(trim_and_cast<T>, 0, stream, n_output_elements, mlp->m_padded_output_width, mlp->m_output_width, output.data(), _output.data());
  }

  // clang-format on
}

#endif

void 
tcnn_inference(network_t handler, cudaStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<float>& output)
{
#if 1
  try {
    handler->inference(stream, input, output);
  }
  catch (std::runtime_error& e) {
    std::cerr << e.what() << std::endl;
    return;
  }
#else

#ifdef ENABLE_IN_SHADER

  constexpr int N_POS_DIMS = TCNN_N_POS_DIMS;
  constexpr int N_FEATURES_PER_LEVEL = 8;
  constexpr int WIDTH = 64;

  using EncodingType = GridEncodingTemplated<precision_t, N_POS_DIMS, N_FEATURES_PER_LEVEL>;
  using NetworkType  = FullyFusedMLP<precision_t, WIDTH>;

  auto encoding = dynamic_cast<EncodingType*>(handler->m_encoding.get());
  assert(encoding && "wrong encoding type");

  auto network = dynamic_cast<NetworkType*>(handler->m_network.get());
  assert(network && "wrong network type");

  // auto encoding_output_layout = encoding->output_layout();
  // encoding->set_output_layout(CM); // use column major encoding here

#if 0

  DeviceNeuralVolume<precision_t, N_POS_DIMS, N_FEATURES_PER_LEVEL, WIDTH> nn(encoding, network);
  nn.batch_sample(stream, input, output);

#else

  GPUMatrixDynamic<precision_t> network_input = {
    encoding->num_encoded_dims(), input.n(), stream, encoding->output_layout()
  };

#if 0
  encoding->encode(stream, input.n(), { input.data(), input.m() }, { network_input.data(), network_input.m() }, nullptr, true);
#else
  DeviceNeuralEncoder<precision_t, N_POS_DIMS, N_FEATURES_PER_LEVEL> d_encoder(encoding);
  d_encoder.batch_encode(stream, input.n(), { input.data(), input.m() }, { network_input.data(), network_input.m() });
#endif

#if 0
  network->inference(stream, network_input, output);
#else
  DeviceNeuralNetwork<precision_t, WIDTH> d_network(network);
  d_network.batch_inference(stream, network_input, output);
#endif

#endif

  // encoding->set_output_layout(encoding_output_layout); // restore output layout

#endif

#endif
}

#ifdef ENABLE_IN_SHADER

template struct DeviceNeuralVolume<precision_t, TCNN_N_POS_DIMS, 1, 16>;
template struct DeviceNeuralVolume<precision_t, TCNN_N_POS_DIMS, 2, 16>;
template struct DeviceNeuralVolume<precision_t, TCNN_N_POS_DIMS, 4, 16>;
template struct DeviceNeuralVolume<precision_t, TCNN_N_POS_DIMS, 8, 16>;

template struct DeviceNeuralVolume<precision_t, TCNN_N_POS_DIMS, 1, 32>;
template struct DeviceNeuralVolume<precision_t, TCNN_N_POS_DIMS, 2, 32>;
template struct DeviceNeuralVolume<precision_t, TCNN_N_POS_DIMS, 4, 32>;
template struct DeviceNeuralVolume<precision_t, TCNN_N_POS_DIMS, 8, 32>;

template struct DeviceNeuralVolume<precision_t, TCNN_N_POS_DIMS, 1, 64>;
template struct DeviceNeuralVolume<precision_t, TCNN_N_POS_DIMS, 2, 64>;
template struct DeviceNeuralVolume<precision_t, TCNN_N_POS_DIMS, 4, 64>;
template struct DeviceNeuralVolume<precision_t, TCNN_N_POS_DIMS, 8, 64>;

#endif

}
} // namespace vnr
