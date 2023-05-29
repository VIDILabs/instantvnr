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

template<typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, bool ROW_MAJOR_OUTPUT>
__global__ void
DeviceNeuralEncoder_batch_encode_kernel(
  const uint32_t num_elements,
  const DeviceNeuralEncoder<T, N_POS_DIMS, N_FEATURES_PER_LEVEL> self,
  const float* __restrict__ coords, // <- column major
  T* __restrict__ encoded_coords)   // <- column or row major
{
  using namespace TCNN_NAMESPACE;

  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_elements) return;

  const uint32_t level = blockIdx.y;

  // CHANGE: now each thread is handling the encoding independently
  // const uint32_t level = blockIdx.y; // <- the level is the same for all threads

  // CHANGE: assume 'max_level_gpu' is always null
  // if (max_level_gpu) {
  //   max_level = (max_level_gpu[i] * num_grid_features) / N_FEATURES_PER_LEVEL;
  // } else {
  //   max_level = (max_level * num_grid_features) / N_FEATURES_PER_LEVEL;
  // }

  float coord[N_POS_DIMS];
  T encoded_coord_per_level[N_FEATURES_PER_LEVEL];

  // CHANGE: do not use pitched pointer
  // #pragma unroll
  // for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) { input[dim] = (float)_input(i)[dim]; }

  #pragma unroll
  for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) { coord[dim] = (float)coords[dim + i * N_POS_DIMS]; }

  // for (uint32_t level = 0; level < self.num_levels; ++level) 
  {
    self.encode_one_level(level, coord, encoded_coord_per_level);

    // COMMENT: essentially doing concatenation
    if (ROW_MAJOR_OUTPUT) {
      #pragma unroll
      for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
        encoded_coords[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = encoded_coord_per_level[f];
      }
    }
    else {
      #pragma unroll
      for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
        encoded_coords[(level * N_FEATURES_PER_LEVEL + f) + i * self.num_grid_features] = encoded_coord_per_level[f];
      }
    }
  }
}

// template<typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, bool ROW_MAJOR_OUTPUT>
// __global__ void
// dynamic_parallel_nested(
//   const uint32_t num_elements, const uint32_t offset,
//   const DeviceNeuralEncoder<T, N_POS_DIMS, N_FEATURES_PER_LEVEL> self,
//   const float* __restrict__ coords, // <- column major
//   T* __restrict__ encoded_coords)   // <- column or row major
// {
//   using namespace TCNN_NAMESPACE;
// 
//   const uint32_t i = offset + threadIdx.x;
//   if (i >= num_elements) return;
//   const uint32_t level = blockIdx.x;
// 
//   float coord[N_POS_DIMS];
//   #pragma unroll
//   for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) { coord[dim] = (float)coords[dim + i * N_POS_DIMS]; }
// 
//   T encoded_coord_per_level[N_FEATURES_PER_LEVEL];
//   self.encode_one_level(level, coord, encoded_coord_per_level);
// 
//   if (ROW_MAJOR_OUTPUT) {
//     #pragma unroll
//     for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
//       encoded_coords[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = encoded_coord_per_level[f];
//     }
//   }
//   else {
//     #pragma unroll
//     for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
//       encoded_coords[(level * N_FEATURES_PER_LEVEL + f) + i * self.num_grid_features] = encoded_coord_per_level[f];
//     }
//   }
// }

// template<typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, bool ROW_MAJOR_OUTPUT>
// __global__ void
// dynamic_parallel_kernel(
//   const uint32_t num_elements,
//   const DeviceNeuralEncoder<T, N_POS_DIMS, N_FEATURES_PER_LEVEL> self,
//   const float* __restrict__ coords, // <- column major
//   T* __restrict__ encoded_coords)   // <- column or row major
// {
//   using namespace TCNN_NAMESPACE;
// 
//   const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
//   if (i >= num_elements) return;
// 
//   if (threadIdx.x == 0) {
//     cudaStream_t s;
//     cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
//     dynamic_parallel_nested<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, ROW_MAJOR_OUTPUT>
//       <<<self.num_levels-1, blockDim.x, 0, s>>>(num_elements, blockIdx.x * blockDim.x, self, coords, encoded_coords);
//   }
// 
//   // do level = self.num_levels-1
//   float coord[N_POS_DIMS];
//   #pragma unroll
//   for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) { coord[dim] = (float)coords[dim + i * N_POS_DIMS]; }
// 
//   uint32_t level = self.num_levels-1;
// 
//   T encoded_coord_per_level[N_FEATURES_PER_LEVEL];
//   self.encode_one_level(level, coord, encoded_coord_per_level);
// 
//   if (ROW_MAJOR_OUTPUT) {
//     #pragma unroll
//     for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
//       encoded_coords[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = encoded_coord_per_level[f];
//     }
//   }
//   else {
//     #pragma unroll
//     for (uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) {
//       encoded_coords[(level * N_FEATURES_PER_LEVEL + f) + i * self.num_grid_features] = encoded_coord_per_level[f];
//     }
//   }
// }

template<typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL>
void
DeviceNeuralEncoder<T, N_POS_DIMS, N_FEATURES_PER_LEVEL>::batch_encode(cudaStream_t stream,
                                                                       const uint32_t num_elements,
                                                                       PitchedPtr<const float> inputs,
                                                                       PitchedPtr<T> outputs)
{
  using namespace TCNN_NAMESPACE;

  if (self->m_n_padded_output_dims == 0 || num_elements == 0) { return; }

  // CHANGE: this is not used anymore
  // GPUMemory<float>* positions = &self->m_positions[stream];
  // positions->enlarge(num_elements * N_POS_DIMS);

  // CHANGE: do not create 'synced_streams'
  // SyncedMultiStream synced_streams{stream, m_n_to_pad > 0 ? 2u : 1u};

  // COMMENT: take care of padding on the auxiliary stream
  // CHANGE: using the input stream instead of 'synced_streams.get(1)'
  if (self->m_n_to_pad > 0) { // clang-format off
    parallel_for_gpu_aos(stream, num_elements, self->m_n_to_pad,
                         [n_output_dims = self->m_n_output_dims, outputs] __device__(size_t elem, size_t dim) {
                           outputs(elem)[n_output_dims + dim] = 0;
                         });
  } // clang-format on

  // COMMENT: first step: extract positional input into consecutive memory for fast reading
  // CHANGE: this step is skipped
  // const dim3 threads = { 64, N_POS_DIMS, 1 };
  // const uint32_t blocks = div_round_up(num_elements, threads.x);
  // extract_position<float, N_POS_DIMS><<<blocks, threads, 0, synced_streams.get(0)>>>(
  //   num_elements, inputs, positions->data()
  // );

  // COMMENT: compute encoding
  // CHANGE: each thread is now handling encodings independently, so do not specify them
  static constexpr uint32_t N_THREADS_HASHGRID = 512;
  // const dim3 blocks_hashgrid = { div_round_up(num_elements, N_THREADS_HASHGRID), 1, 1 };
  const dim3 blocks_hashgrid = { div_round_up(num_elements, N_THREADS_HASHGRID), self->m_n_levels, 1 };

  // T* rm_encoded_positions = outputs.ptr;
  // GPUMemoryArena::Allocation workspace;
  //
  // if (self->m_output_layout == CM) {
  //   workspace = allocate_workspace(stream, num_elements * self->m_n_features * sizeof(T));
  //   rm_encoded_positions = (T*)workspace.data();
  // }

  // CHANGE: use a customized kernel
  // kernel_grid<T, N_POS_DIMS, N_FEATURES_PER_LEVEL>
  //   <<<blocks_hashgrid, N_THREADS_HASHGRID, 0, synced_streams.get(0)>>>(
  //     num_elements,
  //     m_n_features,
  //     m_hashmap_offsets_table.data(),
  //     m_base_resolution,
  //     std::log2(m_per_level_scale),
  //     this->m_quantize_threshold,
  //     this->m_max_level,
  //     this->m_max_level_gpu,
  //     m_interpolation_type,
  //     m_grid_type,
  //     is_inference ? m_grid_inference : m_grid,
  //     positions->data(),
  //     rm_encoded_positions,
  //     dy_dx);

  DeviceNeuralEncoder<T, N_POS_DIMS, N_FEATURES_PER_LEVEL> encoder(self);
  if (self->m_output_layout == RM) {
    DeviceNeuralEncoder_batch_encode_kernel<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, true>
      <<<blocks_hashgrid, N_THREADS_HASHGRID, 0, stream>>>(num_elements, encoder, inputs.ptr, outputs.ptr);
  }
  else {
    DeviceNeuralEncoder_batch_encode_kernel<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, false>
      <<<blocks_hashgrid, N_THREADS_HASHGRID, 0, stream>>>(num_elements, encoder, inputs.ptr, outputs.ptr);
  }

  // if (self->m_output_layout == CM) {
  //   // third step: transpose result (was stored row major due to coalescing)
  //   const dim3 threads_transpose = { self->m_n_levels * N_FEATURES_PER_LEVEL, 8, 1 };
  //   const uint32_t blocks_transpose = div_round_up(num_elements, threads_transpose.y);
  //   transpose_encoded_position<T><<<blocks_transpose, threads_transpose, 0, stream>>>(
  //     num_elements, rm_encoded_positions, outputs);
  // }
}

template struct DeviceNeuralEncoder<precision_t, TCNN_N_POS_DIMS, 1>;
template struct DeviceNeuralEncoder<precision_t, TCNN_N_POS_DIMS, 2>;
template struct DeviceNeuralEncoder<precision_t, TCNN_N_POS_DIMS, 4>;
template struct DeviceNeuralEncoder<precision_t, TCNN_N_POS_DIMS, 8>;

}
}
