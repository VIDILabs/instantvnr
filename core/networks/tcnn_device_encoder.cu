#include "tcnn_device_api.h"

/* namespace instant neural volume */
namespace vnr {
namespace tcnn_impl {

template<typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, HashType HASH_TYPE>
__device__ void
encode(const EncoderCtx<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, HASH_TYPE>& ctx, 
       const uint32_t level, const float* __restrict__ input, T* __restrict__ output)
{
  using namespace TCNN_NAMESPACE;

  if (level >= ((ctx.max_level * ctx.num_grid_features) / N_FEATURES_PER_LEVEL) + 1e-3f) return;

  const T* __restrict__ grid = ctx.grid + ctx.offset_table.data[level] * N_FEATURES_PER_LEVEL;
  const uint32_t hashmap_size = ctx.offset_table.data[level + 1] - ctx.offset_table.data[level];

  const float scale = grid_scale(level, ctx.log2_per_level_scale, ctx.base_resolution);
  const uint32_t resolution = grid_resolution(scale);

  float pos[N_POS_DIMS];
  uint32_t pos_grid[N_POS_DIMS];

  if (ctx.interpolation_type == InterpolationType::Nearest || ctx.interpolation_type == InterpolationType::Linear) {
    #pragma unroll
    for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
      pos_fract(input[dim], &pos[dim], &pos_grid[dim], scale, identity_fun);
    }
  } else {
    #pragma unroll
    for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
      pos_fract(input[dim], &pos[dim], &pos_grid[dim], scale, smoothstep);
    }
  }

  const auto grid_val = [&](const uint32_t local_pos[N_POS_DIMS]) {
    const uint32_t index = grid_index<N_POS_DIMS, HASH_TYPE>(ctx.grid_type, hashmap_size, resolution, local_pos) * N_FEATURES_PER_LEVEL;
    return *(vector_t<T, N_FEATURES_PER_LEVEL>*)&grid[index];
  };

  if (ctx.interpolation_type == InterpolationType::Nearest) {
    *(vector_t<T, N_FEATURES_PER_LEVEL>*)output = grid_val(pos_grid);
    return;
  }

  // N-linear interpolation
  vector_t<T, N_FEATURES_PER_LEVEL> result = {};

  #pragma unroll
  for (uint32_t idx = 0; idx < (1 << N_POS_DIMS); ++idx) {
    float weight = 1;
    uint32_t pos_grid_local[N_POS_DIMS];

    #pragma unroll
    for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
      if ((idx & (1<<dim)) == 0) {
        weight *= 1 - pos[dim];
        pos_grid_local[dim] = pos_grid[dim];
      } else {
        weight *= pos[dim];
        pos_grid_local[dim] = pos_grid[dim] + 1;
      }
    }

    auto val = grid_val(pos_grid_local);

    #pragma unroll
    for (uint32_t feature = 0; feature < N_FEATURES_PER_LEVEL; ++feature) {
      float data = (float)((T*)&val)[feature];
      if (fabsf(data) < ctx.quantize_threshold) data = 0.f;
      ((T*)&result)[feature] += (T)(weight * data);
    }
  }

  // Write to local array instead
  *(vector_t<T, N_FEATURES_PER_LEVEL>*)output = result;
}

#define template_instantiation(T, N_POS_DIMS, HASH_TYPE) \
template __device__ void encode(const EncoderCtx<T, N_POS_DIMS, 1, HASH_TYPE>& ctx, const uint32_t level, const float* __restrict__ input, T* __restrict__ output); \
template __device__ void encode(const EncoderCtx<T, N_POS_DIMS, 2, HASH_TYPE>& ctx, const uint32_t level, const float* __restrict__ input, T* __restrict__ output); \
template __device__ void encode(const EncoderCtx<T, N_POS_DIMS, 4, HASH_TYPE>& ctx, const uint32_t level, const float* __restrict__ input, T* __restrict__ output); \
template __device__ void encode(const EncoderCtx<T, N_POS_DIMS, 8, HASH_TYPE>& ctx, const uint32_t level, const float* __restrict__ input, T* __restrict__ output);

template_instantiation(precision_t, VNR_INPUT_DIMS, HashType::Prime);
template_instantiation(precision_t, VNR_INPUT_DIMS, HashType::CoherentPrime);
template_instantiation(precision_t, VNR_INPUT_DIMS, HashType::ReversedPrime);
// template_instantiation(precision_t, VNR_INPUT_DIMS, HashType::Rng);

}
}
