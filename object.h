//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //

/**
 * Geometry Types Defined by the Application
 */
#ifndef OVR_VOLUME_H
#define OVR_VOLUME_H

#include "core/instantvnr_types.h"
#include "core/network.h"

#include <colormap.h>

#include <cuda_runtime.h>

#include <cassert>
#include <cstring>
#include <limits>
#include <sstream>
#include <vector>

namespace vnr {

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

template<typename T>
struct HasSbtEquivalent {
private:
  CUDABuffer sbt_buffer;
  mutable char* sbt_data{ NULL };
  mutable size_t sbt_size{ 0 };

protected:
  T self;

public:
  virtual ~HasSbtEquivalent()
  {
    sbt_buffer.free(/*stream=*/nullptr);
  }

  virtual CUdeviceptr get_sbt_pointer(cudaStream_t stream) = 0;

  virtual void commit(cudaStream_t stream) = 0;

  void UpdateSbtData(cudaStream_t stream) { sbt_buffer.upload_async(sbt_data, sbt_size, stream); }

  CUdeviceptr CreateSbtPtr(cudaStream_t stream)
  {
    sbt_data = (char*)&self;
    sbt_size = sizeof(T);

    /* create and upload to GPU */
    sbt_buffer.alloc_and_upload_async(&self, 1, stream);
    return sbt_buffer.d_pointer();
  }

  CUdeviceptr GetSbtPtr() const { return sbt_buffer.d_pointer(); }
};

struct InstantiableGeometry {
  affine3f matrix;

  InstantiableGeometry() { matrix = affine3f(gdt::one); }

  /*! compute 3x4 transformation matrix */
  void transform(float transform[12]) const;
};

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

struct StructuredRegularVolume
  : protected HasSbtEquivalent<DeviceVolume>
  , public InstantiableGeometry {
private:
  range1f original_data_range;

  std::vector<vec4f> colors_data;
  std::vector<float> alphas_data;

  float sampling_rate = 1.f;

  cudaArray_t tfn_color_array_handler{};
  cudaArray_t tfn_alpha_array_handler{};

public:
  ~StructuredRegularVolume();
  StructuredRegularVolume();

  CUdeviceptr get_sbt_pointer(cudaStream_t stream) override;
  void commit(cudaStream_t stream) override;

  DeviceVolume* d_pointer() const { return (DeviceVolume*)GetSbtPtr(); }

  const vec3i& get_dims() const { return self.volume.dims; }
  const float& get_sampling_rate() const { return sampling_rate; }

  void set_volume(Array3DScalar& v);
  void set_volume(cudaTextureObject_t data);
  void set_volume(cudaTextureObject_t data, ValueType type, vec3i dims, range1f original_data_range);
  void set_clipping(vec3f lower, vec3f upper);
  void set_macrocell(vec3i dims, vec3f spacings, vec2f* d_value_range, float* d_max_opacity);
  void set_transfer_function(cudaStream_t stream, const std::vector<vec3f>& c, const std::vector<vec2f>& o, const range1f& r);
  void set_sampling_rate(float r);
  void set_density_scale(float scale);
  bool empty() const { return self.volume.data == 0; }

  DeviceVolume& device() { return self; }
};

} // namespace ovr
#endif // OVR_VOLUME_H
