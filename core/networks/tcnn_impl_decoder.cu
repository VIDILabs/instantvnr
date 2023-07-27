#include "tcnn_device_api.h"

/* namespace instant neural volume */
namespace vnr {
namespace tcnn_impl {

template<typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL>
__device__ void
DeviceNeuralEncoder<T, N_POS_DIMS, N_FEATURES_PER_LEVEL>::encode_one_level(
  const uint32_t level,            // <- the same for all threads
  const float* __restrict__ input, // <- local array float[N_POS_DIMS]
  T* __restrict__ output_per_level // <- local array T [N_FEATURES_PER_LEVEL]
) const
{
  using namespace TCNN_NAMESPACE;

  if (level >= max_level + 1e-3f) {
    #pragma unroll
    for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
      // CHANGES: write to local array instead
      // encoded_positions[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = (T)0.0f;
      output_per_level[f] = (T)0.0f;
    }

    // CHANGE: remove gradient calculation
    // Gradient is zero for zeroed-out dimensions.
    // if (dy_dx) {
    //   #pragma unroll
    //   for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
    //     ((vector_fullp_t<N_POS_DIMS>*)dy_dx)[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = {0};
    //   }
    // }

    return;
  }

  auto* __restrict__ grid = grid_data; // CHANGE: access grid data in terms of a member variable
  grid += hashmap_offset_table[level] * N_FEATURES_PER_LEVEL;
  const uint32_t hashmap_size = hashmap_offset_table[level + 1] - hashmap_offset_table[level];

  const float scale = exp2f(level * log2_per_level_scale) * base_resolution - 1.0f;
  const uint32_t grid_resolution = ((uint32_t)std::ceil(scale) + 1);

  float pos[N_POS_DIMS];
  // float pos_derivative[N_POS_DIMS]; // CHANGE: remove gradient calculation
  uint32_t pos_grid[N_POS_DIMS];

  if (interpolation_type == InterpolationType::Nearest || interpolation_type == InterpolationType::Linear) {
    #pragma unroll
    for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
      // CHANGE: access position differently, using the non-gradient version
      // pos_fract(positions_in[i + dim * num_elements], &pos[dim], &pos_derivative[dim], &pos_grid[dim],
      //           scale, identity_fun, identity_derivative);
      pos_fract(input[dim], &pos[dim], &pos_grid[dim], scale, identity_fun);
    }
  }
  else {
    #pragma unroll
    for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
      // CHANGE: access position differently, using the non-gradient version
      // pos_fract(positions_in[i + dim * num_elements], &pos[dim], &pos_derivative[dim], &pos_grid[dim],
      //           scale, smoothstep, smoothstep_derivative);
      pos_fract(input[dim], &pos[dim], &pos_grid[dim], scale, smoothstep);
    }
  }

  auto grid_value = [&](const uint32_t local_pos[N_POS_DIMS]) -> PerLevelVec {
    const uint32_t index =
      grid_index<N_POS_DIMS, N_FEATURES_PER_LEVEL>(grid_type, 0, hashmap_size, grid_resolution, local_pos);
    return *(PerLevelVec*)&grid[index];
  };

  if (interpolation_type == InterpolationType::Nearest) {
    auto result = grid_value(pos_grid);

    #pragma unroll
    for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
      // CHANGES: write to local array instead
      // encoded_positions[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = result[f];
      output_per_level[f] = result[f];
    }

    // CHANGE: remove gradient calculation
    // Gradient is zero when there's no interpolation.
    // if (dy_dx) {
    //   #pragma unroll
    //   for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
    //     ((vector_fullp_t<N_POS_DIMS>*)dy_dx)[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = {0};
    //   }
    // }

    return;
  }

  // N-linear interpolation
  PerLevelVec result = {};

  #pragma unroll
  for (uint32_t idx = 0; idx < (1 << N_POS_DIMS); ++idx) {
    float weight = 1;
    uint32_t pos_grid_local[N_POS_DIMS];

    #pragma unroll
    for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
      if ((idx & (1 << dim)) == 0) {
        weight *= 1 - pos[dim];
        pos_grid_local[dim] = pos_grid[dim];
      }
      else {
        weight *= pos[dim];
        pos_grid_local[dim] = pos_grid[dim] + 1;
      }
    }

    auto value = grid_value(pos_grid_local);

    #pragma unroll
    for (uint32_t feature = 0; feature < N_FEATURES_PER_LEVEL; ++feature) {
      float data = (float)((T*)&value)[feature];
      if (fabsf(data) < quantize_threshold) data = 0.f;
      ((T*)&result)[feature] += (T)(weight * data);
    }
  }

  #pragma unroll
  for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
    // CHANGES: write to local array instead
    // encoded_positions[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = result[f];
    output_per_level[f] = result[f];
  }

  // CHANGE: remove gradient calculation
  // Gradient
  // if (dy_dx) {
  //   vector_fullp_t<N_POS_DIMS> grads[N_FEATURES_PER_LEVEL] = {};
  //
  //   #pragma unroll
  //   for (uint32_t grad_dim = 0; grad_dim < N_POS_DIMS; ++grad_dim) {
  //     #pragma unroll
  //     for (uint32_t idx = 0; idx < (1 << (N_POS_DIMS-1)); ++idx) {
  //       float weight = scale;
  //       uint32_t pos_grid_local[N_POS_DIMS];
  //
  //       #pragma unroll
  //       for (uint32_t non_grad_dim = 0; non_grad_dim < N_POS_DIMS-1; ++non_grad_dim) {
  //         const uint32_t dim = non_grad_dim >= grad_dim ? (non_grad_dim+1) : non_grad_dim;
  //
  //         if ((idx & (1<<non_grad_dim)) == 0) {
  //           weight *= 1 - pos[dim];
  //           pos_grid_local[dim] = pos_grid[dim];
  //         } else {
  //           weight *= pos[dim];
  //           pos_grid_local[dim] = pos_grid[dim] + 1;
  //         }
  //       }
  //
  //       pos_grid_local[grad_dim] = pos_grid[grad_dim];
  //       auto val_left = grid_val(pos_grid_local);
  //       pos_grid_local[grad_dim] = pos_grid[grad_dim] + 1;
  //       auto val_right = grid_val(pos_grid_local);
  //
  //       #pragma unroll
  //       for (uint32_t feature = 0; feature < N_FEATURES_PER_LEVEL; ++feature) {
  //         grads[feature][grad_dim] +=
  //          weight * ((float)val_right[feature] - (float)val_left[feature]) * pos_derivative[grad_dim];
  //       }
  //     }
  //   }
  //
  //   #pragma unroll
  //   for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
  //     ((vector_fullp_t<N_POS_DIMS>*)dy_dx)[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = grads[f];
  //   }
  // }
}

template struct DeviceNeuralEncoder<precision_t, TCNN_N_POS_DIMS, 1>;
template struct DeviceNeuralEncoder<precision_t, TCNN_N_POS_DIMS, 2>;
template struct DeviceNeuralEncoder<precision_t, TCNN_N_POS_DIMS, 4>;
template struct DeviceNeuralEncoder<precision_t, TCNN_N_POS_DIMS, 8>;

}
}
