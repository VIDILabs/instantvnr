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

#include "types.h"

#include "network.h"

#include <colormap.h>

#include <cuda_runtime.h>

#include <cassert>
#include <cstring>
#include <limits>
#include <sstream>
#include <vector>

namespace vnr {

// inline std::pair<Array1DFloat4, cudaArray_t>
// CreateColorMap(const std::string& name)
// {
//   if (colormap::data.count(name) > 0) {
//     std::vector<vec4f>& arr = *((std::vector<vec4f>*)colormap::data.at(name));
//     return CreateArray1DFloat4(arr);
//   }
//   else {
//     throw std::runtime_error("Unexpected colormap name: " + name);
//   }
// }

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

#if defined(ENABLE_OPTIX)

#define ALIGN_SBT __align__(OPTIX_SBT_RECORD_ALIGNMENT)

/*! SBT record for a raygen program */
struct ALIGN_SBT RaygenRecord {
  ALIGN_SBT char header[OPTIX_SBT_RECORD_HEADER_SIZE]{};
  // just a dummy value - later examples will use more interesting data here
  void* data{};
};

/*! SBT record for a miss program */
struct ALIGN_SBT MissRecord {
  ALIGN_SBT char header[OPTIX_SBT_RECORD_HEADER_SIZE]{};
  // just a dummy value - later examples will use more interesting data here
  void* data{};
};

/*! SBT record for a hitgroup program */
struct ALIGN_SBT HitgroupRecord {
  ALIGN_SBT char header[OPTIX_SBT_RECORD_HEADER_SIZE]{};
  void* data{};
};

OptixTraversableHandle
buildas_exec(OptixDeviceContext context, cudaStream_t stream, std::vector<OptixBuildInput> input, CUDABuffer& asBuffer);

#endif

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

  virtual CUdeviceptr d_pointer() const = 0;

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

struct AabbGeometry {
private:
#if defined(ENABLE_OPTIX)
  // the AABBs for procedural geometries
  OptixAabb aabb{ 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };
#endif

  CUDABuffer aabbBuffer;
  CUDABuffer asBuffer; // buffer that keeps the (final, compacted) accel structure

public:
  ~AabbGeometry()
  {
    aabbBuffer.free(0);
    asBuffer.free(0);
  }

#if defined(ENABLE_OPTIX)
  OptixTraversableHandle buildas(OptixDeviceContext optixContext, cudaStream_t stream = 0);
#endif
};

struct MeshGeometry {
private:
  uint32_t asBuildflag{};
  CUDABuffer asBuffer; // buffer that keeps the (final, compacted) accel structure

protected:
  /*! the model we are going to trace rays against */
  std::vector<vec3f> vertex;
  std::vector<vec3i> index;

  /*! one buffer per input mesh */
  CUDABuffer vertexBuffer;
  CUDABuffer indexBuffer;

public:
  ~MeshGeometry()
  {
    asBuffer.free(0);
    vertexBuffer.free(0);
    indexBuffer.free(0);
  }

#if defined(ENABLE_OPTIX)
  OptixTraversableHandle buildas(OptixDeviceContext optixContext, cudaStream_t stream = 0);
#endif
};

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

struct BoxesGeometry
  : protected HasSbtEquivalent<DeviceBoxes>
  , public MeshGeometry
  , public InstantiableGeometry {
private:
  uint64_t num_boxes{ 0 };

  // CUDABuffer colorBuffer;

private:
  void add_unit_cube(const affine3f& xfm)
  {
    int firstVertexID = (int)vertex.size();
    vertex.push_back(xfmPoint(xfm, vec3f(0.f, 0.f, 0.f)));
    vertex.push_back(xfmPoint(xfm, vec3f(1.f, 0.f, 0.f)));
    vertex.push_back(xfmPoint(xfm, vec3f(0.f, 1.f, 0.f)));
    vertex.push_back(xfmPoint(xfm, vec3f(1.f, 1.f, 0.f)));
    vertex.push_back(xfmPoint(xfm, vec3f(0.f, 0.f, 1.f)));
    vertex.push_back(xfmPoint(xfm, vec3f(1.f, 0.f, 1.f)));
    vertex.push_back(xfmPoint(xfm, vec3f(0.f, 1.f, 1.f)));
    vertex.push_back(xfmPoint(xfm, vec3f(1.f, 1.f, 1.f)));

    int indices[] = { 0, 1, 3, 2, 3, 0, 5, 7, 6, 5, 6, 4, 0, 4, 5, 0, 5, 1,
                      2, 3, 7, 2, 7, 6, 1, 5, 7, 1, 7, 3, 4, 0, 2, 4, 2, 6 };
    for (int i = 0; i < 12; i++)
      index.push_back(firstVertexID + vec3i(indices[3 * i + 0], indices[3 * i + 1], indices[3 * i + 2]));
  }

public:
  CUdeviceptr get_sbt_pointer(cudaStream_t stream) override;
  CUdeviceptr d_pointer() const override { return GetSbtPtr(); }
  void commit(cudaStream_t stream) override;

  void add_cube(const vec3f& center, const vec3f& size)
  {
    affine3f xfm;
    xfm.p = center - 0.5f * size;
    xfm.l.vx = vec3f(size.x, 0.f, 0.f);
    xfm.l.vy = vec3f(0.f, size.y, 0.f);
    xfm.l.vz = vec3f(0.f, 0.f, size.z);
    add_unit_cube(xfm);
    ++num_boxes;
  }

  // vec3f* color_buffer_ptr() const { return (vec3f*)colorBuffer.d_pointer(); }
};

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

struct StructuredRegularVolume
  : protected HasSbtEquivalent<DeviceVolume>
  , public AabbGeometry
  , public InstantiableGeometry {
private:
  range1f original_data_range;

  std::vector<vec4f> colors_data;
  std::vector<float> alphas_data;

  float sampling_rate = 1.f;

  // cudaArray_t volume_array_handler{};
  cudaArray_t tfn_color_array_handler{};
  cudaArray_t tfn_alpha_array_handler{};

public:
  ~StructuredRegularVolume();

  CUdeviceptr get_sbt_pointer(cudaStream_t stream) override;
  CUdeviceptr d_pointer() const override { return GetSbtPtr(); }
  void commit(cudaStream_t stream) override;

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

// struct BlockBrickVolume {};

} // namespace ovr
#endif // OVR_VOLUME_H
