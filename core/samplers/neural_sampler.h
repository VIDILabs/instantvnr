// ----------------------------------------------------------------------------
//  neural_sampler.h
//
//  Concrete `SamplerAPI` variants used for training a neural volume:
//
//    DummySampler                      no data; placeholder used when the
//                                      network is loaded from disk and no
//                                      ground-truth is available.
//    CudaSampler                       single-timestep CUDA 3D texture
//                                      sampler (default GPU path).
//    CudaSampler_TimeVarying           like `CudaSampler` but cycles
//                                      through a list of timestep files.
//    OpenVKLSampler (+GT variant)      OpenVKL-backed sampling; enabled
//                                      when `ENABLE_OPENVKL`. The "GT"
//                                      variant retains the dense grid so
//                                      PSNR/SSIM remain meaningful.
//    OutOfCoreSampler                  mmap + libaio streaming reader for
//                                      volumes larger than GPU memory.
//                                      Enabled when `ENABLE_OUT_OF_CORE`.
//    VirtualMemorySampler              mmap + on-demand upload variant
//                                      used when AIO is unavailable or the
//                                      working set fits in host RAM.
//
//  The header also declares shared helpers: `load_regular_grid`,
//  `normalize_regular_grid`, random-buffer utilities (`random_hbuffer_*`,
//  `random_dbuffer_*`), and `generate_grid_coords` for building training
//  batches on the GPU.
// ----------------------------------------------------------------------------

#pragma once

#include "../sampler.h"

#include <vidi_filemap.h>

#define SAMPLE_WITH_TRILINEAR_INTERPOLATION 1

// The C++ random number generates a better random distribution. However, we want to match our GPU implementation here for experiments.
#define USE_GPU_RANDOM_NUMBER_GENERSTOR 1

namespace vnr {

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

void random_hbuffer_uniform(float* h_buffer, size_t batch);
void random_dbuffer_uniform(float* d_buffer, size_t batch, cudaStream_t stream);

void random_hbuffer_uint32(uint32_t* h_buffer, size_t batch, uint32_t min, uint32_t max);
void random_dbuffer_uint32(uint32_t* d_buffer, size_t batch, uint32_t min, uint32_t max, cudaStream_t stream);

void random_hbuffer_uint64(uint64_t* h_buffer, size_t batch, uint64_t min, uint64_t max);
void random_dbuffer_uint64(uint64_t* d_buffer, size_t batch, uint64_t min, uint64_t max, cudaStream_t stream);

inline void random_hbuffer_uint32(uint32_t* h_buffer, size_t batch, uint32_t count) {
  ASSERT_THROW(count != 0, "calling 'random_hbuffer_uint32' with zero range.");
  random_hbuffer_uint32(h_buffer, batch, 0, count - 1); 
}

inline void random_dbuffer_uint32(uint32_t* d_buffer, size_t batch, uint32_t count, cudaStream_t stream) {
  ASSERT_THROW(count != 0, "calling 'random_dbuffer_uint32' with zero range.");
  random_dbuffer_uint32(d_buffer, batch, 0, count - 1, stream); 
}

inline void random_hbuffer_uint64(uint64_t* h_buffer, size_t batch, uint64_t count) { 
  ASSERT_THROW(count != 0, "calling 'random_hbuffer_uint64' with zero range.");
  random_hbuffer_uint64(h_buffer, batch, 0, count - 1); 
}

inline void random_dbuffer_uint64(uint64_t* d_buffer, size_t batch, uint64_t count, cudaStream_t stream) {
  ASSERT_THROW(count != 0, "calling 'random_dbuffer_uint64' with zero range.");
  random_dbuffer_uint64(d_buffer, batch, 0, count - 1, stream); 
}

void generate_grid_coords(float* d_coords, vec3i grid_origin, vec3i grid_dims, vec3f grid_spacing, cudaStream_t stream);


// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

void load_regular_grid(
  const MultiVolume::File& file, 
  vec3i dims, ValueType type, range1f minmax,
  std::shared_ptr<char[]>& buffer,
  range1f& value_range_unnormalized, 
  range1f& value_range_normalized
);

void normalize_regular_grid(
  std::shared_ptr<char[]>& buffer,
  vec3i dims, ValueType type, range1f minmax,
  range1f& value_range_unnormalized, 
  range1f& value_range_normalized
);

struct DummySampler : SamplerAPI
{
private:
  vec3i m_dims{};
  dtype m_type = VALUE_TYPE_FLOAT;
  range1f m_value_range_normalized = range1f(0.f, 1.f);

public:
  DummySampler(vec3i dims) : m_dims(dims) { }
  cudaTextureObject_t texture() const override { return 0; }
  dtype type() const override { return m_type; }
  vec3i dims() const override { return m_dims; }
  float lower() const override { return m_value_range_normalized.lower; }
  float upper() const override { return m_value_range_normalized.upper; }
  void set_current_volume_timestamp(int index) override { }
  void sample(void* d_coords, void* d_values, size_t num_samples, const vec3f& lower, const vec3f& upper, cudaStream_t stream) override { }
};

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

struct CudaSampler : SamplerAPI
{
protected:
  std::shared_ptr<char[]> m_current_data;

  vec3i m_dims{};
  dtype m_type{};
  cudaTextureObject_t m_texture{0};
  cudaArray_t m_array{NULL};
  range1f m_value_range_normalized;
  range1f m_value_range_unnormalized;

public:
  ~CudaSampler();
  CudaSampler(const void* data, vec3i dims, dtype type, range1f range, bool create_cuda_texture);
  CudaSampler(const MultiVolume::File& file, vec3i dims, dtype type, range1f range, bool create_cuda_texture, bool save_volume_to_debug);
  void* data(int timestamp) const { assert(timestamp == 0); return m_current_data.get(); }
  cudaTextureObject_t texture() const override { return m_texture; }
  dtype type() const override { return m_type; }
  vec3i dims() const override { return m_dims; }
  float lower() const override { return m_value_range_normalized.lower; }
  float upper() const override { return m_value_range_normalized.upper; }
  void sample(void* d_coords, void* d_values, size_t num_samples, const vec3f& lower, const vec3f& upper, cudaStream_t stream);
  void sample_grid(void* d_coords, void* d_values, vec3i grid_origin, vec3i grid_dims, vec3f grid_spacing, cudaStream_t stream);
  void sample_inputs(const void* d_coords, void* d_values, size_t num_samples, cudaStream_t stream);
  void set_current_volume_timestamp(int index) override { if (index != 0) throw std::runtime_error("not implemented"); }
};

struct CudaSampler_TimeVarying : CudaSampler
{
private:
  int m_timestamp = 0;
  std::vector<std::shared_ptr<char[]>> m_dataset;

public:
  CudaSampler_TimeVarying(const MultiVolume& desc, bool save_volume, bool skip_texture);
  void* data(int timestamp) const { return m_dataset[timestamp].get(); }
  void set_current_volume_timestamp(int index) override;
};

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

#ifdef ENABLE_OPENVKL

struct OpenVKLSampler : SamplerAPI
{
protected:
  void* m_volume{};
  void* m_sampler{};

  vec3i m_dims{};
  dtype m_type = VALUE_TYPE_FLOAT;
  box3f m_bbox;

  bool  m_cell_centered = true;

  std::vector<vec3f> m_coords; // for staging data temporarily
  std::vector<float> m_values;

  range1f m_value_range{0.f, 1.f};

  std::shared_ptr<char[]> m_current_data;

public:
  OpenVKLSampler(const std::string& example); // Example Unstructured Volume Loader
  OpenVKLSampler(const std::string& filename, const std::string& field); // VDB Loader
  OpenVKLSampler(const MultiVolume::File& file, vec3i dims, dtype type, range1f range); // Regular Grid Loader
  void create(vec3i dims, dtype _type, range1f range);
  cudaTextureObject_t texture() const override { return 0; }
  dtype type() const override { return m_type; }
  vec3i dims() const override { return m_dims; }
  float lower() const override { return m_value_range.lower; }
  float upper() const override { return m_value_range.upper; }
  void sample(void* d_input, void* d_output, size_t num_samples, const vec3f& lower, const vec3f& upper, cudaStream_t stream) override;
  void sample_grid(void* d_coords, void* d_values, vec3i grid_origin, vec3i grid_dims, vec3f grid_spacing, cudaStream_t stream) override;
  void sample_with_inputs(const vec3f* h_input, float* h_output, size_t num_samples, cudaStream_t stream);
  void set_current_volume_timestamp(int index) override {
    if (index != 0) throw std::runtime_error("OpenVKL sampler do not support timevarying data");
  }
};

struct OpenVKLSampler_WithGroundTruthData : OpenVKLSampler
{
private:
  cudaTextureObject_t m_texture;
  cudaArray_t m_array;

public:
  OpenVKLSampler_WithGroundTruthData(const MultiVolume& desc); // regular grid
  OpenVKLSampler_WithGroundTruthData(const MultiVolume& desc, vec3i downsampled_dims); // downsampled regular grid
  cudaTextureObject_t texture() const override { return m_texture; }
};

#endif // ENABLE_OPENVKL



// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

#ifdef ENABLE_OUT_OF_CORE

struct RandomBuffer;

struct OutOfCoreSampler : SamplerAPI
{
private:
  dtype m_type{};
  vec3i m_dims{};
  range1f m_value_range;
  size_t m_offset{};
  vidi::FileMap m_reader;
  std::shared_ptr<RandomBuffer> m_randbuf;

  std::vector<vec3f> m_coords; // for staging data temporarily
  std::vector<float> m_values;
  std::vector<float> m_random_bidx;
  std::vector<float> m_random_vidx;

public:
  OutOfCoreSampler(const MultiVolume& desc);

  cudaTextureObject_t texture() const override { return 0; }
  dtype type() const override { return m_type; }
  vec3i dims() const override { return m_dims; }
  float lower() const override { return 0.f; }
  float upper() const override { return 1.f; }
  void sample(void* d_coords, void* d_values, size_t num_samples, const vec3f& lower, const vec3f& upper, cudaStream_t stream) override;
  void sample_grid(void* d_coords, void* d_values, vec3i grid_origin, vec3i grid_dims, vec3f grid_spacing, cudaStream_t stream) override;
};

struct VirtualMemorySampler : SamplerAPI
{
private:
  dtype m_type{};
  vec3i m_dims{};
  range1f m_value_range;
  size_t m_offset{};
  vidi::FileMap m_reader;

  std::vector<vec3f> m_coords; // for staging data temporarily
  std::vector<float> m_values;

  vec3f m_fdims;
  float m_value_scale;
  int m_elem_size;

public:
  VirtualMemorySampler(const MultiVolume& desc);

  cudaTextureObject_t texture() const override { return 0; }
  dtype type() const override { return m_type; }
  vec3i dims() const override { return m_dims; }
  float lower() const override { return 0.f; }
  float upper() const override { return 1.f; }
  void sample(void* d_input, void* d_output, size_t num_samples, const vec3f& lower, const vec3f& upper, cudaStream_t stream) override;
  void sample_grid(void* d_coords, void* d_values, vec3i grid_origin, vec3i grid_dims, vec3f grid_spacing, cudaStream_t stream) override;
};

#endif // ENABLE_OUT_OF_CORE

} // namespace vnr
