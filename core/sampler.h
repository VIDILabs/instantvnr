#pragma once

#include "types.h"

#include <memory>

namespace vnr {

struct SamplerAPI // SamplerImpl
{
  typedef ValueType dtype;

  virtual ~SamplerAPI() = default;
  virtual void set_current_volume_timestamp(int index) { if (index != 0) throw std::runtime_error("only support single timestep volume"); }
  virtual cudaTextureObject_t texture() const = 0;
  virtual dtype type() const = 0;
  virtual vec3i dims() const = 0;
  virtual float lower() const = 0;
  virtual float upper() const = 0;
  virtual void sample(void* d_coords, void* d_values, size_t num_samples, const vec3f& lower, const vec3f& upper, cudaStream_t stream) = 0;
  virtual void sample_grid(void* d_coords, void* d_values, vec3i grid_origin, vec3i grid_dims, vec3f grid_spacing, cudaStream_t stream) {};
};

struct Sampler
{
private:
  using dtype = SamplerAPI::dtype;
  std::shared_ptr<SamplerAPI> impl;
  vec3i m_dims{};
  affine3f m_transform;

public:
  Sampler() {}

  float lower() const { return impl->lower(); }
  float upper() const { return impl->upper(); }
  dtype type() const { return impl->type(); }
  cudaTextureObject_t texture() const { return impl->texture(); }

  vec3i dims() const { return m_dims; }
  affine3f transform() const { return m_transform; }

  void set_transform(const affine3f& xfm) { m_transform = xfm; }

  void set_current_volume_index(int index)
  {
    impl->set_current_volume_timestamp(index);
  }

  void take_samples(void* d_input, void* d_output, size_t num_samples, cudaStream_t stream, const vec3f& lower, const vec3f& upper) const
  {
    impl->sample(d_input, d_output, num_samples, lower, upper, stream);
  }

  void take_samples_grid(void* d_input, void* d_output, vec3i grid_origin, vec3i grid_dims, vec3f grid_spacing, cudaStream_t stream) const
  {
    impl->sample_grid(d_input, d_output, grid_origin, grid_dims, grid_spacing, stream);
  }

  void load(const MultiVolume& desc, std::string training_mode, bool save_volume = false);
};

struct VolumeObject 
{
  virtual const cudaTextureObject_t& texture()  const = 0;
  virtual ValueType get_data_type()             const = 0;
  virtual range1f   get_data_value_range()      const = 0;
  virtual vec3i     get_data_dims()             const = 0;
  virtual affine3f  get_data_transform()        const = 0;
  virtual float*    get_macrocell_max_opacity() const = 0;
  virtual vec2f*    get_macrocell_value_range() const = 0;
  virtual vec3i     get_macrocell_dims()        const = 0;
  virtual vec3f     get_macrocell_spacings()    const = 0;
  virtual void set_transfer_function(const std::vector<vec3f>& c, const std::vector<vec2f>& o, const range1f& r) = 0;
  virtual void set_data_transform(affine3f transform) = 0;
};

struct TransferFunctionObject {
  DeviceTransferFunction tfn;
  cudaArray_t tfn_color_array_handler{};
  cudaArray_t tfn_alpha_array_handler{};
  void clean();
  void set_transfer_function(const std::vector<vec3f>& c, const std::vector<vec2f>& o, const range1f& r, cudaStream_t stream);
};

struct MacroCell {
private:
  CUDABuffer m_max_opacity_buffer;
  CUDABuffer m_value_range_buffer;
  vec3i m_volume_dims;
  vec3i m_dims;
  vec3f m_spacings;
  bool m_is_external = false;

public:
  bool allocated() const { return m_max_opacity_buffer.sizeInBytes > 0 && m_value_range_buffer.sizeInBytes > 0; }
  float* d_max_opacity() const { return (float*)m_max_opacity_buffer.d_pointer(); }
  vec2f* d_value_range() const { return (vec2f*)m_value_range_buffer.d_pointer(); }

  vec3i dims()     const { return m_dims; }
  vec3f spacings() const { return m_spacings; }

  bool is_external() const { return m_is_external; }

  void set_dims(vec3i dims) { m_dims = dims; }
  void set_spacings(vec3f spacings) { m_spacings = spacings; }

  void set_shape(vec3i volume_dims);
  void set_external(MacroCell& external);
  void allocate();

  void compute_everything(cudaTextureObject_t volume);
  void update_explicit(vec3f* d_coords, float* d_values, size_t count, cudaStream_t stream);
  void update_implicit(); // NOT IMPLEMENTED

  void update_max_opacity(const DeviceTransferFunction& tfn, cudaStream_t stream);
};

struct SimpleVolume : VolumeObject {
private:
  MultiVolume desc; // make it private!
  cudaTextureObject_t tex;
  TransferFunctionObject tfn;

public:
  Sampler sampler;
  MacroCell macrocell;
  std::string mode;

  void load(const MultiVolume& descriptor, std::string sampling_mode, bool save_volume = false);

  uint32_t get_num_timesteps() const { return desc.data.size(); }
  void set_current_timestep(int index);

  // common API
  const cudaTextureObject_t& texture()  const override { return tex; }
  ValueType get_data_type()             const override { return sampler.type(); }
  range1f   get_data_value_range()      const override { return range1f(sampler.lower(), sampler.upper()); }
  vec3i     get_data_dims()             const override { return sampler.dims(); }
  affine3f  get_data_transform()        const override { return sampler.transform(); }
  float*    get_macrocell_max_opacity() const override { return macrocell.d_max_opacity(); }
  vec2f*    get_macrocell_value_range() const override { return macrocell.d_value_range(); }
  vec3i     get_macrocell_dims()        const override { return macrocell.dims(); }
  vec3f     get_macrocell_spacings()    const override { return macrocell.spacings(); }
  void set_transfer_function(const std::vector<vec3f>& c, const std::vector<vec2f>& o, const range1f& r) override;
  void set_data_transform(affine3f transform) override;
};

}
