//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //

#include "tcnn_network.h"

#ifdef ENABLE_IN_SHADER
#include "tcnn_device_api.h"
#include "tcnn_threadblock.h"
#endif

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

template<typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t WIDTH, HashType HASH_TYPE>
__device__ T DeviceNeuralVolume<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, WIDTH, HASH_TYPE>::sample(const float3 coordinate) const {
  typedef T OUT_T;

  const uint32_t in_width = enc.num_grid_features;

  constexpr uint32_t N_BLOCKS = WIDTH / 16;
  constexpr uint32_t shmem_size_coord  = sizeof(float) * (16 * N_ITERS) * N_POS_DIMS;
  constexpr uint32_t shmem_size_output = sizeof(OUT_T) * (16 * N_ITERS) * 16;

  static_assert(N_POS_DIMS == 3, "N_POS_DIMS must be 3 for volumes");
  static_assert(2 * N_BLOCKS == N_ITERS, "this has to be true: 2 * N_BLOCKS == N_ITERS");

  // Shared memory contains the intermediate activations of blockDim.y*16 elements.
  // In some cases, it also contains the weight matrix for the first and last layer.
  extern __shared__ __half shmem[];

  float* pos_shmem = (float*)shmem;
  OUT_T* out_shmem = shmem + shmem_size_coord / sizeof(__half);
  __half* act_shmem = shmem + (shmem_size_coord + shmem_size_output) / sizeof(__half);

  // Each block computes exactly one 16-element chunk of the batch.
	const uint32_t elem_idx = 16 * blockIdx.x * N_ITERS;

  const __half* __restrict__ weights = mlp.weights;
  const auto ACTIVATION = mlp.activation;
  const auto output_activation = mlp.output_activation;
  const auto n_hidden_matmuls  = mlp.n_hidden_matmuls; 

  OUT_T out[16] = {};

  threadblock_read_pos_from_shmem<N_POS_DIMS, WIDTH, N_ITERS>(pos_shmem, (float3&)coordinate);

  // First layer
  if (in_width != WIDTH) {
    threadblock_input_layer_forward_dynamic_with_encoding<N_POS_DIMS, N_FEATURES_PER_LEVEL, WIDTH, N_ITERS, OUT_T, nvcuda::wmma::row_major>(ACTIVATION, act_shmem, weights, in_width, 0, pos_shmem, enc);
  } else {
    threadblock_load_input_static_with_encoding<N_POS_DIMS, N_FEATURES_PER_LEVEL, WIDTH, N_ITERS>(act_shmem, pos_shmem, enc);
    threadblock_layer<WIDTH, N_ITERS, OUT_T>(ACTIVATION, act_shmem, weights, nullptr);
  }

	const uint32_t first_weights_stride = WIDTH * in_width;
	const uint32_t weights_stride = WIDTH * WIDTH;

  // Hidden layers
  for (uint32_t k = 0; k < n_hidden_matmuls; ++k) {
    threadblock_layer<WIDTH, N_ITERS, OUT_T>(ACTIVATION, act_shmem, weights + first_weights_stride + weights_stride * k, nullptr);
  }

  // Last layer
  threadblock_last_layer_forward<WIDTH, N_ITERS, OUT_T>(output_activation, act_shmem, weights + first_weights_stride + weights_stride * n_hidden_matmuls, out_shmem, 16, nvcuda::wmma::mem_row_major);

  threadblock_write_output_from_shmem<WIDTH, N_ITERS>(out_shmem, out);

  static_assert(VNR_OUTPUT_DIMS == 1, "non-scalar neural representation not fully supported in the in-shader path.");
  return out[0];
}

#define template_instantiation(T, N_POS_DIMS, WIDTH) \
template struct DeviceNeuralVolume<T, N_POS_DIMS, 1, WIDTH>; \
template struct DeviceNeuralVolume<T, N_POS_DIMS, 2, WIDTH>; \
template struct DeviceNeuralVolume<T, N_POS_DIMS, 4, WIDTH>; \
template struct DeviceNeuralVolume<T, N_POS_DIMS, 8, WIDTH>;

template_instantiation(precision_t, VNR_INPUT_DIMS, 16);
template_instantiation(precision_t, VNR_INPUT_DIMS, 32);
template_instantiation(precision_t, VNR_INPUT_DIMS, 64);

#endif

}
} // namespace vnr
