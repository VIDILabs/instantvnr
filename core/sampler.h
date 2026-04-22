// ----------------------------------------------------------------------------
//  sampler.h
//
//  `SamplerAPI`  - abstract host-side interface for every way the library
//                  can read voxel values during training:
//                    * sample()      draw a batch at arbitrary positions
//                    * sample_grid() evaluate on a regular subgrid
//                    * texture()     expose a CUDA texture of the source
//                                    volume (when one exists)
//                  Concrete implementations live under `core/samplers/`
//                  (CudaSampler, OpenVKLSampler, OutOfCoreSampler, ...).
//
//  `SamplerAPI::create(desc, mode, save_volume)` is the string-dispatched
//  factory wired into `api.cpp`. Supported `mode` strings: "GPU", "NOTHING",
//  plus "VIRTUAL_MEMORY" / "OUT_OF_CORE" (ENABLE_OUT_OF_CORE) and several
//  "OPENVKL*" variants (ENABLE_OPENVKL).
//
//  `SimpleVolume` is the ground-truth volume wrapper built on top of
//  `SamplerAPI`. It owns the sampler, the CUDA volume texture, a
//  `MacroCell` for empty-space skipping, and a per-volume
//  `TransferFunctionObject`. It implements the `VolumeObject` interface
//  from `instantvnr_types.h` so that simple and neural volumes are
//  interchangeable at the renderer boundary.
// ----------------------------------------------------------------------------

#pragma once

#include "instantvnr_types.h"
#include "macrocell.h"

#include <memory>

namespace vnr {

struct SamplerAPI // SamplerImpl
{
private:
  vec3i m_rendering_dims{};
  affine3f m_transform;

public:
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

  vec3i rendering_dims() const { return m_rendering_dims; }
  void set_rendering_dims(const vec3i& dims) { m_rendering_dims = dims; }
  affine3f transform() const { return m_transform; }
  void set_transform(const affine3f& xfm) { m_transform = xfm; }
  void set_current_volume_index(int index) { set_current_volume_timestamp(index); }
  void take_samples(void* d_input, void* d_output, size_t num_samples, cudaStream_t stream, const vec3f& lower, const vec3f& upper) {
    sample(d_input, d_output, num_samples, lower, upper, stream);
  }
  void take_samples_grid(void* d_input, void* d_output, vec3i grid_origin, vec3i grid_dims, vec3f grid_spacing, cudaStream_t stream) {
    sample_grid(d_input, d_output, grid_origin, grid_dims, grid_spacing, stream);
  }

  static std::shared_ptr<SamplerAPI> 
  create(const MultiVolume& desc, std::string training_mode, bool save_volume = false);
};

typedef std::shared_ptr<SamplerAPI> Sampler;

struct SimpleVolume : VolumeObject {
private:
  MultiVolume desc; // make it private!
  cudaTextureObject_t tex;
  TransferFunctionObject tfn;

public:
  Sampler sampler;
  MacroCell macrocell;
  std::string mode;

  void load(const void* data, vec3i dims, std::string type, range1f range, std::string sampling_mode);
  void load(const MultiVolume& descriptor, std::string sampling_mode, bool save_volume = false);

  uint32_t get_num_timesteps() const { return (uint32_t)desc.data.size(); }
  void set_current_timestep(int index);

  // common API
  const cudaTextureObject_t& texture()  const override { return tex; }
  ValueType get_data_type()             const override { return sampler->type(); }
  range1f   get_data_value_range()      const override { return range1f(sampler->lower(), sampler->upper()); }
  vec3i     get_data_dims()             const override { return sampler->rendering_dims(); }
  affine3f  get_data_transform()        const override { return sampler->transform(); }
  float*    get_macrocell_max_opacity() const override { return macrocell.d_max_opacity(); }
  vec2f*    get_macrocell_value_range() const override { return macrocell.d_value_range(); }
  vec3i     get_macrocell_dims()        const override { return macrocell.dims(); }
  vec3f     get_macrocell_spacings()    const override { return macrocell.spacings(); }
  void set_transfer_function(const std::vector<vec3f>& c, const std::vector<vec2f>& o, const range1f& r) override;
  void set_data_transform(affine3f transform) override;
};

}
