#include "neural_sampler.h"

#include <tiny-cuda-nn/random.h>
using TCNN_NAMESPACE :: generate_random_uniform;
using default_rng_t = TCNN_NAMESPACE :: default_rng_t;

namespace vidi {
enum VoxelType {
  VOXEL_UINT8  = vnr::VALUE_TYPE_UINT8,
  VOXEL_INT8   = vnr::VALUE_TYPE_INT8,
  VOXEL_UINT16 = vnr::VALUE_TYPE_UINT16,
  VOXEL_INT16  = vnr::VALUE_TYPE_INT16,
  VOXEL_UINT32 = vnr::VALUE_TYPE_UINT32,
  VOXEL_INT32  = vnr::VALUE_TYPE_INT32,
  VOXEL_FLOAT  = vnr::VALUE_TYPE_FLOAT,
  VOXEL_DOUBLE = vnr::VALUE_TYPE_DOUBLE,
};
} // namespace vidi
#define VIDI_VOLUME_EXTERNAL_TYPE_ENUM
#include <vidi_parallel_algorithm.h>
#include <vidi_volume_reader.h>

#include <tbb/parallel_for.h>

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

namespace {

template<class T>
constexpr const T& clamp(const T& v, const T& lo, const T& hi)
{
    return (v < lo) ? lo : (hi < v) ? hi : v;
}

template<typename IType>
std::shared_ptr<char[]>
convert_volume(std::shared_ptr<char[]> idata, size_t size, float vmin, float vmax)
{
  std::shared_ptr<char[]> odata;
  odata.reset(new char[size * sizeof(float)]);

  tbb::parallel_for(size_t(0), size, [&](size_t idx) {
    auto* i = (IType*)&idata[idx * sizeof(IType)];
    auto* o = (float*)&odata[idx * sizeof(float)];
#ifdef TEST_SIREN
    *o = clamp((static_cast<float>(*i) - (float)vmin) / ((float)vmax - (float)vmin), 0.f, 1.f) * 2.f - 1.f;
#else
    *o = clamp((static_cast<float>(*i) - (float)vmin) / ((float)vmax - (float)vmin), 0.f, 1.f);
#endif
  });

  return odata;
}

template<>
std::shared_ptr<char[]>
convert_volume<float>(std::shared_ptr<char[]> idata, size_t size, float vmin, float vmax)
{
  tbb::parallel_for(size_t(0), size, [&](size_t idx) {
    auto* i = (float*)&idata[idx * sizeof(float)];
#ifdef TEST_SIREN
    *i = clamp((static_cast<float>(*i) - (float)vmin) / ((float)vmax - (float)vmin), 0.f, 1.f) * 2.f - 1.f;
#else
    *i = clamp((static_cast<float>(*i) - (float)vmin) / ((float)vmax - (float)vmin), 0.f, 1.f);
#endif
  });

  return idata;
}

template<typename T>
static range1f
compute_scalar_fminmax(const void* _array, size_t count)
{
  using vidi::parallel::compute_scalar_minmax;
  auto r = compute_scalar_minmax<T>(_array, count, 0);
  return range1f((float)r.first, (float)r.second);
}

} // namespace

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

__global__ void 
generate_coords(uint32_t n_elements, vec3i lower, vec3i size, vec3f rdims, float* __restrict__ coords);

StaticSampler::~StaticSampler()
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

StaticSampler::StaticSampler(vec3i dims, ValueType type)
{
  m_dims = dims;
  m_type = VALUE_TYPE_FLOAT;
  m_value_range_normalized.lower = 0.f;
  m_value_range_normalized.upper = 1.f;

  m_value_range_unnormalized = m_value_range_normalized;
}

StaticSampler::StaticSampler(const MultiVolume& desc, bool save_volume, bool skip_texture)
{
  const auto& dims = desc.dims;
  const auto& type = desc.type;
  const auto& range = desc.range;

  m_dims = dims;
  m_type = VALUE_TYPE_FLOAT;
  m_dataset.resize(desc.data.size());

  load(desc.data[0], dims, type, range, 
       m_dataset[0], 
       m_value_range_unnormalized, 
       m_value_range_normalized);

#if 1 /* save volume */
  if (save_volume) {
    vidi::FileMap w = vidi::filemap_write_create("reference.bin", sizeof(float) * dims.long_product());
    vidi::filemap_random_write(w, 0, (float*)m_dataset[0].get(), sizeof(float) * dims.long_product());
    vidi::filemap_close(w);
    log() << "[vnr] saved the reference volume to: reference.bin" << std::endl;
  }
#endif

  // generate a texture to represent the ground truth
  if (!skip_texture) {
    CreateArray3DScalar<float>(m_array, m_texture, dims, SAMPLE_WITH_TRILINEAR_INTERPOLATION, (float*)m_dataset[0].get());
  }

  for (int i = 1; i < desc.data.size(); ++i) {
    range1f unnormalized, normalized;
    load(desc.data[i], dims, type, range, m_dataset[i], unnormalized, normalized);
    m_value_range_unnormalized.extend(unnormalized);
    m_value_range_normalized.extend(normalized);
  }
}

void
StaticSampler::load(const MultiVolume::File& desc,
                    vec3i dims, dtype type, range1f minmax,
                    std::shared_ptr<char[]>& buffer,
                    range1f& value_range_unnormalized,
                    range1f& value_range_normalized)
{
  const auto& offset = desc.offset;
  const auto& filename = desc.filename;
  const auto& is_big_endian = desc.bigendian;

  /* load data from file */
  {
    vidi::StructuredRegularVolumeDesc desc;
    desc.dims.x = dims.x;
    desc.dims.y = dims.y;
    desc.dims.z = dims.z;
    desc.type = (vidi::VoxelType)type;
    desc.offset = offset;
    desc.is_big_endian = is_big_endian;
    buffer = vidi::read_volume_structured_regular(filename, desc);
  }

  /* copy data to GPU */
  const size_t count = (size_t)dims.x * dims.y * dims.z;

  /* convert volume into floats */
  range1f range;
  {
    if (minmax.is_empty()) {
      switch (type) {
      case VALUE_TYPE_UINT8: range = compute_scalar_fminmax<uint8_t>(buffer.get(), count); break;
      case VALUE_TYPE_INT8: range = compute_scalar_fminmax<int8_t>(buffer.get(), count); break;
      case VALUE_TYPE_UINT16: range = compute_scalar_fminmax<uint16_t>(buffer.get(), count); break;
      case VALUE_TYPE_INT16: range = compute_scalar_fminmax<int16_t>(buffer.get(), count); break;
      case VALUE_TYPE_UINT32: range = compute_scalar_fminmax<uint32_t>(buffer.get(), count); break;
      case VALUE_TYPE_INT32: range = compute_scalar_fminmax<int32_t>(buffer.get(), count); break;
      case VALUE_TYPE_FLOAT: range = compute_scalar_fminmax<float>(buffer.get(), count); break;
      case VALUE_TYPE_DOUBLE: range = compute_scalar_fminmax<double>(buffer.get(), count); break;
      default: throw std::runtime_error("unknown data type");
      }
    }
    else {
      range = minmax;
    }

    switch (type) {
    case VALUE_TYPE_UINT8: buffer = convert_volume<uint8_t>  (buffer, count, range.lower, range.upper); break;
    case VALUE_TYPE_INT8: buffer = convert_volume<int8_t>    (buffer, count, range.lower, range.upper); break;
    case VALUE_TYPE_UINT16: buffer = convert_volume<uint16_t>(buffer, count, range.lower, range.upper); break;
    case VALUE_TYPE_INT16: buffer = convert_volume<int16_t>  (buffer, count, range.lower, range.upper); break;
    case VALUE_TYPE_UINT32: buffer = convert_volume<uint32_t>(buffer, count, range.lower, range.upper); break;
    case VALUE_TYPE_INT32: buffer = convert_volume<int32_t>  (buffer, count, range.lower, range.upper); break;
    case VALUE_TYPE_FLOAT: buffer = convert_volume<float>    (buffer, count, range.lower, range.upper); break;
    case VALUE_TYPE_DOUBLE: buffer = convert_volume<double>  (buffer, count, range.lower, range.upper); break;
    default: throw std::runtime_error("unknown data type");
    }
  }
  value_range_unnormalized = range;
  value_range_normalized.lower = 0.f;
  value_range_normalized.upper = 1.f;
  // std::tie(value_range_normalized.lower, value_range_normalized.upper) = vidi::parallel::compute_scalar_minmax<float>(buffer.get(), count, 0);

  log() << "[vnr] unnormalized range " << value_range_unnormalized.lower << " " << value_range_unnormalized.upper << std::endl;
  log() << "[vnr] normalized range " << value_range_normalized.lower << " " << value_range_normalized.upper << std::endl;
}

void 
StaticSampler::set_current_volume_timestamp(int index)
{
  CopyLinearMemoryToArray<float>(m_dataset[index].get(), m_array, m_dims, cudaMemcpyHostToDevice);
  m_timestamp = index;
}

void
StaticSampler::sample(void* d_input, void* d_output, size_t batch_size, const vec3f& lower, const vec3f& upper, cudaStream_t stream)
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
StaticSampler::sample_grid(void* d_coords, void* d_values, vec3i grid_origin, vec3i grid_dims, vec3f grid_spacing, cudaStream_t stream)
{
  generate_grid_coords((float*)d_coords, grid_origin, grid_dims, grid_spacing, stream);
  sample_inputs(d_coords, d_values, grid_dims.long_product(), stream);
}

void 
StaticSampler::sample_inputs(const void* d_coords, void* d_values, size_t num_samples, cudaStream_t stream)
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

} // namespace vnr
