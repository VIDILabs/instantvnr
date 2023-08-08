#include "neural_sampler.h"

#include <tiny-cuda-nn/random.h>
using TCNN_NAMESPACE :: generate_random_uniform;
using default_rng_t = TCNN_NAMESPACE :: default_rng_t;

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

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

template<typename T>
void normalize_buffer_device(const void* data, vec3i dims, range1f range, CUDABufferTyped<float>& d_floats, void *h_buffer = nullptr)
{
  const size_t count = (size_t)dims.x * (size_t)dims.y * (size_t)dims.z; // copy raw data to GPU

  CUDABuffer d_buffer;
  d_buffer.alloc_and_upload_async((T*)data, count, NULL);

  double vmin;
  double scale;
  if (range.is_empty()) {
    const auto d_ptr = thrust::device_ptr<T>((T*)d_buffer.d_pointer());
    T value_max = thrust::reduce(d_ptr, d_ptr + count, std::numeric_limits<T>::min(), thrust::maximum<T>());
    T value_min = thrust::reduce(d_ptr, d_ptr + count, std::numeric_limits<T>::max(), thrust::minimum<T>());
    vmin = (double)value_min;
    scale = 1.0 / ((double)value_max - (double)value_min);
  }
  else {
    vmin = range.lower;
    scale = 1.0 / (range.upper - range.lower);
  }

  // for now, we convert everything to floats
  d_floats.alloc(count*sizeof(float), NULL);
  util::parallel_for_gpu(count, [in=(T*)d_buffer.d_pointer(), out=(float*)d_floats.d_pointer(), vmin, scale] __device__ (int64_t i) {
    out[i] = (float)(((double)in[i]  - vmin) * scale);
  });

  d_buffer.free();

  if (h_buffer) {
    CUDA_CHECK(cudaMemcpy(h_buffer, d_floats.d_pointer(), count * sizeof(float), cudaMemcpyDeviceToHost));
  }
}

CudaSampler::~CudaSampler()
{
  if (m_array) {
    CUDA_CHECK_NOEXCEPT(cudaFreeArray(m_array));
    util::total_n_bytes_allocated() -= m_dims.long_product() * sizeof(float);
    m_array = NULL;
  }

  if (m_texture) {
    CUDA_CHECK_NOEXCEPT(cudaDestroyTextureObject(m_texture));
    m_texture = 0;
  }
}

CudaSampler::CudaSampler(const void* data, vec3i dims, dtype type, range1f range, bool create_cuda_texture)
  : m_dims(dims)
  , m_type(VALUE_TYPE_FLOAT)
{
  // we dont use this data handler for sampling, but we keep it for other functionalities
  const size_t count = (size_t)dims.x * (size_t)dims.y * (size_t)dims.z;
  m_current_data.reset(new char[count * value_type_size(type)]);
  // m_current_data.reset((char *)data, [](char*) { /* does not own the data */ });

  CUDABufferTyped<float> d_floats;
  switch (type) {
  case VALUE_TYPE_UINT8:  normalize_buffer_device< uint8_t>(data, dims, range, d_floats, m_current_data.get()); break;
  case VALUE_TYPE_INT8:   normalize_buffer_device<  int8_t>(data, dims, range, d_floats, m_current_data.get()); break;
  case VALUE_TYPE_UINT16: normalize_buffer_device<uint16_t>(data, dims, range, d_floats, m_current_data.get()); break;
  case VALUE_TYPE_INT16:  normalize_buffer_device< int16_t>(data, dims, range, d_floats, m_current_data.get()); break;
  case VALUE_TYPE_UINT32: normalize_buffer_device<uint32_t>(data, dims, range, d_floats, m_current_data.get()); break;
  case VALUE_TYPE_INT32:  normalize_buffer_device< int32_t>(data, dims, range, d_floats, m_current_data.get()); break;
  case VALUE_TYPE_UINT64: normalize_buffer_device<uint64_t>(data, dims, range, d_floats, m_current_data.get()); break;
  case VALUE_TYPE_INT64:  normalize_buffer_device< int64_t>(data, dims, range, d_floats, m_current_data.get()); break;
  case VALUE_TYPE_FLOAT:  normalize_buffer_device<   float>(data, dims, range, d_floats, m_current_data.get()); break;
  case VALUE_TYPE_DOUBLE: normalize_buffer_device<  double>(data, dims, range, d_floats, m_current_data.get()); break;
  default: throw std::runtime_error("unsupported data type");
  }

  // generate a texture to represent the ground truth
  if (create_cuda_texture) {
    assert(!m_array);
    assert(!m_texture);
    CreateArray3DScalar<float>(m_array, m_texture, dims, SAMPLE_WITH_TRILINEAR_INTERPOLATION); // create an empty texture
    CopyLinearMemoryToArray<float>((float*)d_floats.d_pointer(), m_array, dims, cudaMemcpyDeviceToDevice);
  }

  d_floats.free();
}

CudaSampler::CudaSampler(const MultiVolume::File& file, vec3i dims, dtype type, range1f range, bool create_cuda_texture, bool save_volume_to_debug)
  : m_dims(dims)
  , m_type(VALUE_TYPE_FLOAT)
{
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
    assert(!m_array);
    assert(!m_texture);
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
