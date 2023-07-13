#pragma once

#include "instantvnr_types.h"

INSTANT_VNR_NAMESPACE_BEGIN

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

INSTANT_VNR_NAMESPACE_END

