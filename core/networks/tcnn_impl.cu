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

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

#ifdef ENABLE_IN_SHADER

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

#endif

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

// template<int N_FEATURES_PER_LEVEL, int WIDTH>
// void tcnn_inference_batch(network_t handler, cudaStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<float>& output);

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
  tcnn_inference_batch<4, 32>(handler, stream, input, output);
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
