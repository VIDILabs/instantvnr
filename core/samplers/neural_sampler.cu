#include "neural_sampler.h"

#include <tiny-cuda-nn/random.h>
using TCNN_NAMESPACE :: generate_random_uniform;
using default_rng_t = TCNN_NAMESPACE :: default_rng_t;

// #define TEST_SIREN 

#ifdef ENABLE_LOGGING
#define log() std::cout
#else
static std::ostream null_output_stream(0);
#define log() null_output_stream
#endif

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

namespace vnr {

// namespace {
// template<class T>
// constexpr const T& clamp(const T& v, const T& lo, const T& hi) { return (v < lo) ? lo : (hi < v) ? hi : v; }
// } // namespace

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

static default_rng_t rng{ 1337 };

void random_dbuffer_uniform(float* d_buffer, size_t batch, cudaStream_t stream)
{
  generate_random_uniform<float>(stream, rng, batch, d_buffer); // [0, 1)
}

void random_dbuffer_uint32(uint32_t* d_buffer, size_t batch, uint32_t min, uint32_t max, cudaStream_t stream)
{
  generate_random_uniform<uint32_t>(stream, rng, batch, d_buffer, min, max); // [min, max)
}

void random_dbuffer_uint64(uint64_t* d_buffer, size_t batch, uint64_t min, uint64_t max, cudaStream_t stream)
{
  generate_random_uniform<uint64_t>(stream, rng, batch, d_buffer, min, max); // [min, max)
}

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

// function defined in network.cu (line 51-68)
__global__ void generate_coords(uint32_t n_elements, vec3i lower, vec3i size, vec3f rdims, float* __restrict__ coords);

CudaSampler::~CudaSampler()
{
  // TODO: cleanup 3D textures
  // if (m_array) {
  //   CUDA_CHECK_NOEXCEPT(cudaFreeArray(m_array));
  //   m_array = NULL;
  //   // util::total_n_bytes_allocated() -= tfn.alphas.length * sizeof(float);
  // }
  // if (data) {
  //   CUDA_CHECK(cudaDestroyTextureObject(data));
  //   data = 0;
  // }
  // dims = 0;
}

CudaSampler::CudaSampler(const MultiVolume::File& file, vec3i dims, dtype type, range1f range, bool create_cuda_texture, bool save_volume_to_debug)
  : m_dims(dims)
  , m_type(VALUE_TYPE_FLOAT)
{
  m_dims = dims;
  m_type = VALUE_TYPE_FLOAT;

  load_regular_grid(file, dims, type, range, 
       m_current_data, 
       m_value_range_unnormalized, 
       m_value_range_normalized
  );

#if 1 /* save volume */
  if (save_volume_to_debug) {
    vidi::FileMap w = vidi::filemap_write_create("reference.bin", sizeof(float) * dims.long_product());
    vidi::filemap_random_write(w, 0, (float*)m_current_data.get(), sizeof(float) * dims.long_product());
    vidi::filemap_close(w);
    log() << "[vnr] saved the reference volume to: reference.bin" << std::endl;
  }
#endif

  // generate a texture to represent the ground truth
  if (create_cuda_texture) {
    CreateArray3DScalar<float>(m_array, m_texture, dims, SAMPLE_WITH_TRILINEAR_INTERPOLATION, (float*)m_current_data.get());
  }
}

void
CudaSampler::sample(void* d_input, void* d_output, size_t batch_size, const vec3f& lower, const vec3f& upper, cudaStream_t stream)
{
  TRACE_CUDA;

  // The C++ random number generates a better random distribution. As a result, the training performs better.
  // However, the GPU based method is likely faster.
#if 0 
  static std::vector<float> coords;
  coords.resize(batch_size*3);
  tbb::parallel_for(size_t(0), batch_size, [&](size_t i) { // cpp 1
    coords[i*3+0] = uniform_random(0.f, 1.f);
    coords[i*3+1] = uniform_random(0.f, 1.f);
    coords[i*3+2] = uniform_random(0.f, 1.f);
  });
  CUDA_CHECK(cudaMemcpyAsync(d_input, coords.data(), sizeof(vec3f) * batch_size, cudaMemcpyHostToDevice, stream));
#else
  random_dbuffer_uniform((float*)d_input, batch_size * 3, stream);
#endif

  util::parallel_for_gpu(0, stream, batch_size, [lower=lower, scale=upper-lower, volume=m_texture, coords=(vec3f*)d_input, values=(float*)d_output] __device__ (size_t i) {
    const auto p = lower + coords[i] * scale;
    coords[i] = p;
    tex3D<float>(values + i, volume, p.x, p.y, p.z);
  });

  // float value_max = -float_large;
  // float value_min = +float_large;
  // const auto gt = thrust::device_ptr<float>((float*)d_output);
  // value_max = thrust::reduce(gt, gt + batch_size, value_max, thrust::maximum<float>());
  // value_min = thrust::reduce(gt, gt + batch_size, value_min, thrust::minimum<float>());
  // printf("min %f, max %f\n", value_min, value_max);

  TRACE_CUDA;
}

void 
CudaSampler::sample_grid(void* d_coords, void* d_values, vec3i grid_origin, vec3i grid_dims, vec3f grid_spacing, cudaStream_t stream)
{
  generate_grid_coords((float*)d_coords, grid_origin, grid_dims, grid_spacing, stream);
  sample_inputs(d_coords, d_values, grid_dims.long_product(), stream);
}

void 
CudaSampler::sample_inputs(const void* d_coords, void* d_values, size_t num_samples, cudaStream_t stream)
{
  // const vec3f lower(0.f);
  // const vec3f scale(1.f);

  util::parallel_for_gpu(0, stream, num_samples, [volume=m_texture, coords=(const vec3f*)d_coords, values=(float*)d_values] __device__ (size_t i) {
    const auto p = coords[i];
    tex3D<float>(values + i, volume, p.x, p.y, p.z);
  });

  // const vec3f hspacing = 0.5f / vec3f(m_dims);
  // util::parallel_for_gpu(0, stream, num_samples, [volume=m_texture, hspacing=hspacing, coords=(vec3f*)input.data(), values=(float*)d_output] __device__ (size_t i) {
  //
  //   const auto p = coords[i];
  //   float v1;
  //   tex3D<float>(&v1, volume, p.x, p.y, p.z);
  //
  //   const auto pos = clamp(coords[i], hspacing, vec3f(1.f) - hspacing);
  //   float v2;
  //   tex3D<float>(&v2, volume, pos.x, pos.y, pos.z);
  //
  //   assert(v1 == v2);
  //   values[i] = v2;
  // });
}


// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

CudaSampler_TimeVarying::CudaSampler_TimeVarying(const MultiVolume& desc, bool save_volume, bool skip_texture)
  : CudaSampler(desc.data[0], desc.dims, desc.type, desc.range, !skip_texture, save_volume)
{
  m_dataset.resize(desc.data.size());
  m_dataset[0] = CudaSampler::m_current_data;
  for (int i = 1; i < desc.data.size(); ++i) {
    range1f unnormalized, normalized;
    load_regular_grid(desc.data[i], desc.dims, desc.type, desc.range, m_dataset[i], unnormalized, normalized);
    m_value_range_unnormalized.extend(unnormalized);
    m_value_range_normalized.extend(normalized);
  }
}

void 
CudaSampler_TimeVarying::set_current_volume_timestamp(int index)
{
  CopyLinearMemoryToArray<float>(m_dataset[index].get(), m_array, m_dims, cudaMemcpyHostToDevice);
  m_timestamp = index;
}

} // namespace vnr
