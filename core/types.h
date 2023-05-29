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
#ifndef OVR_TYPES_H
#define OVR_TYPES_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda/cuda_buffer.h>

#include "mathdef.h"
#include <gdt/random/random.h>

#if defined(ENABLE_OPTIX)
#include <optix.h>
#include <optix_stubs.h>
#endif

#include <array>
#include <vector>
#include <stdio.h>

#ifdef CUDA_BUFFER_VERBOSE_MEMORY_ALLOCS
#define VNR_VERBOSE_MEMORY_ALLOCS
#endif
// #define VNR_VERBOSE_MEMORY_ALLOCS

namespace vnr {

struct MultiVolume {
  struct File {
    std::string filename;
    size_t offset;
    size_t nbytes;
    bool bigendian;
    void* rawptr{ nullptr };
  };

  vec3i     dims;
  ValueType type;
  range1f   range;
  vec3f scale = 0;
  vec3f translate = 0;
  std::vector<File> data;
};

struct TransferFunction {
  std::vector<vec3f> color;
  std::vector<vec2f> alpha;
  range1f range;
};


// ------------------------------------------------------------------
// Array Definitions
// ------------------------------------------------------------------

struct Array1D
{
  ValueType type;
  size_t length{ 0 };
  cudaTextureObject_t data{ 0 }; // the storage of the data on texture unit
  void* rawptr{ nullptr };
};

using Array1DScalar = Array1D;
using Array1DFloat4 = Array1D;

struct Array3DScalar
{
  ValueType type;
  vec3i dims { 0 };
  vec3f rdims{ 0 };
  cudaTextureObject_t data{}; // the storage of the data on texture unit
};

// ------------------------------------------------------------------
// Object Definitions
// ------------------------------------------------------------------

enum ObjectType {
  VOLUME_STRUCTURED_REGULAR,
  GEOMETRY_BOXES,
};

#ifndef __NVCC__
static std::string
object_type_string(ObjectType t)
{
  switch (t) {
  case VOLUME_STRUCTURED_REGULAR: return "VOLUME_STRUCTURED_REGULAR";
  case GEOMETRY_BOXES: return "GEOMETRY_BOXES";
  default: throw std::runtime_error("unknown object type");
  }
}
#endif

enum {
  VISIBILITY_MASK_GEOMETRY = 0x1,
  VISIBILITY_MASK_VOLUME = 0x2,
};

struct DeviceTransferFunction
{
  Array1DFloat4 colors;
  Array1DScalar alphas;
  range1f range;
  float range_rcp_norm;
};

struct DeviceVolume
{
  Array3DScalar volume;
  DeviceTransferFunction tfn;

  float step_rcp = 1.f;
  float step = 1.f;
  float density_scale = 1.f;

  vec3f grad_step;

  float* __restrict__ macrocell_max_opacity{ nullptr };
  vec2f* __restrict__ macrocell_value_range{ nullptr };
  vec3i macrocell_dims;
  vec3f macrocell_spacings;
  vec3f macrocell_spacings_rcp;

  box3f bbox = box3f(vec3f(0), vec3f(1)); // object space box
};

struct DeviceBoxes
{
  vec3f* __restrict__ vertex;
  vec3i* __restrict__ index;
  vec3f* __restrict__ per_box_color;
};

struct DeviceSpheres
{
  vec3f* __restrict__ per_box_color;
};

// ------------------------------------------------------------------
// Shared Definitions
// ------------------------------------------------------------------

struct SciVisMaterial {
  const float ambient;
  const float diffuse;
  const float specular;
  const float shininess;
};

/* shared global data */
struct LaunchParams
{
  struct DeviceFrameBuffer
  {
    vec4f* __restrict__ rgba;
    vec2i size;
  } frame;
  vec4f* __restrict__ accumulation{ nullptr };
  int32_t frame_index{ 0 };

  struct DeviceCamera
  {
    vec3f position;
    vec3f direction;
    vec3f horizontal;
    vec3f vertical;
  } camera, last_camera;

  affine3f transform;

  const float raymarching_shadow_sampling_scale = 2.f;

  /* lights */
  const float scivis_shading_scale = 0.95;

  SciVisMaterial mat_gradient_shading{ .6f, .9f, .4f, 40.f };
  SciVisMaterial mat_full_shadow{ 1.f, .5f, .4f, 40.f };
  SciVisMaterial mat_single_shade_heuristic{ 0.8f, .2f, .4f, 40.f };

  float light_ambient{ 1.5f };
  vec3f light_directional_rgb{ 1.0f };
  vec3f light_directional_dir{ 0.7, 0.9, 0.4 };
};

struct Camera
{
public:
  /*! camera position - *from* where we are looking */
  vec3f from;
  /*! which point we are looking *at* */
  vec3f at;
  /*! general up-vector */
  vec3f up;
  /*! fovy in degrees */
  float fovy = 60;
};

using RandomTEA = gdt::LCG<16>;

#define float_large 1e20f
#define float_small std::numeric_limits<float>::min() // to avoid division by zero
#define float_epsilon std::numeric_limits<float>::epsilon()
#define nearly_one 0.9999f

template <typename T>
__forceinline__ __device__ T lerp(float r, const T& a, const T& b) 
{
  return (1-r) * a + r * b;
}

// ------------------------------------------------------------------
// Helper Functions
// ------------------------------------------------------------------
template<typename Type>
static void // copy from CUDA linear memory to CUDA array
CopyLinearMemoryToArray(void* src_pointer, cudaArray_t dst_array, vec3i dims, cudaMemcpyKind kind)
{
  cudaMemcpy3DParms param = { 0 };
  param.srcPos = make_cudaPos(0, 0, 0);
  param.dstPos = make_cudaPos(0, 0, 0);
  param.srcPtr = make_cudaPitchedPtr(src_pointer, dims.x * sizeof(Type), dims.x, dims.y);
  param.dstArray = dst_array;
  param.extent = make_cudaExtent(dims.x, dims.y, dims.z);
  param.kind = kind;
  CUDA_CHECK(cudaMemcpy3D(&param));
}

template<typename T>
static void
CreateArray3DScalar(cudaArray_t& array, cudaTextureObject_t& tex, vec3i dims, bool trilinear, void* ptr = nullptr)
{
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<T>();
  CUDA_CHECK(cudaMalloc3DArray(&array, &channel_desc, make_cudaExtent(dims.x, dims.y, dims.z)));
  util::total_n_bytes_allocated() += (size_t)dims.x * dims.y * dims.z * sizeof(T);
#ifdef VNR_VERBOSE_MEMORY_ALLOCS
	std::cout << "3D Texture: Allocating " << util::prettyBytes((size_t)dims.x * dims.y * dims.z * sizeof(T)) << std::endl;
#endif
  const auto filter = trilinear ? cudaFilterModeLinear : cudaFilterModePoint;
  tex = createCudaTexture<T>(array, cudaReadModeElementType, filter, filter);
  if (ptr) CopyLinearMemoryToArray<T>((T*)ptr, array, dims, cudaMemcpyHostToDevice);
}

template<typename T> 
static void
CreateArray1DScalar(cudaStream_t stream, const std::vector<T>& input, cudaArray_t& array_handler, Array1DScalar& output)
{
  static_assert(std::is_scalar<T>::value, "expecting a scalar type");

  output.type = value_type<T>();
  output.length = input.size();

  cudaChannelFormatDesc desc; cudaExtent extent{}; unsigned int flags;
  if (array_handler) {
    CUDA_CHECK(cudaArrayGetInfo(&desc, &extent, &flags, array_handler));
  }

  if (extent.width != input.size()) {
    if (array_handler) {
      CUDA_CHECK(cudaFreeArray(array_handler));
      array_handler = 0;
    }
    if (output.data) {
      CUDA_CHECK(cudaDestroyTextureObject(output.data));
      output.data = 0;
    }
    if (output.rawptr) {
      CUDA_CHECK(cudaFree(output.rawptr));
      output.rawptr = 0;
    }
  }

  if (!array_handler) {
    array_handler = allocateCudaArray1D<T>(input.data(), input.size());
  }
  else {
    fillCudaArray1D<T>(array_handler, input.data(), input.size());
  }

  if (!output.data) {
    output.data = createCudaTexture<T>(array_handler, cudaReadModeElementType, cudaFilterModeLinear);
  }

  if (!output.rawptr) {
    CUDA_CHECK(cudaMallocAsync((void**)&output.rawptr, output.length * sizeof(T), stream));
    util::total_n_bytes_allocated() += output.length * sizeof(T);
#ifdef VNR_VERBOSE_MEMORY_ALLOCS
    std::cout << "[mem] Linear " << util::prettyBytes(output.length * sizeof(T)) << std::endl;
#endif
  }
  CUDA_CHECK(cudaMemcpyAsync(output.rawptr, (void*)input.data(), output.length  * sizeof(T), cudaMemcpyHostToDevice, stream));
  // CUDA_SYNC_CHECK();
}

static void
CreateArray1DFloat4(cudaStream_t stream, const std::vector<float4>& input, cudaArray_t& array_handler, Array1DFloat4& output)
{
  output.type = VALUE_TYPE_FLOAT4;
  output.length = input.size();

  cudaChannelFormatDesc desc; cudaExtent extent{}; unsigned int flags;
  if (array_handler) {
    CUDA_CHECK(cudaArrayGetInfo(&desc, &extent, &flags, array_handler));
  }

  if (extent.width != input.size()) {
    if (array_handler) {
      CUDA_CHECK(cudaFreeArray(array_handler));
      array_handler = 0;
    }
    if (output.data) {
      CUDA_CHECK(cudaDestroyTextureObject(output.data));
      output.data = 0;
    }
    if (output.rawptr) {
      CUDA_CHECK(cudaFree(output.rawptr));
      output.rawptr = 0;
    }
  }

  if (!array_handler) {
    array_handler = allocateCudaArray1D<float4>(input.data(), input.size());
  }
  else {
    fillCudaArray1D<float4>(array_handler, input.data(), input.size());
  }

  if (!output.data) {
    output.data = createCudaTexture<float4>(array_handler, cudaReadModeElementType, cudaFilterModeLinear);
  }

  if (!output.rawptr) {
    CUDA_CHECK(cudaMallocAsync((void**)&output.rawptr, output.length * sizeof(float4), stream));
    util::total_n_bytes_allocated() += output.length * sizeof(float4);
#ifdef VNR_VERBOSE_MEMORY_ALLOCS
	  std::cout << "[mem] Linear " << util::prettyBytes(output.length * sizeof(float4)) << std::endl;
#endif
  }
  CUDA_CHECK(cudaMemcpyAsync(output.rawptr, (void*)input.data(), output.length  * sizeof(float4), cudaMemcpyHostToDevice, stream));
  // CUDA_SYNC_CHECK();
}

static void
CreateArray1DFloat4(cudaStream_t stream, const std::vector<vec4f>& input, cudaArray_t& array_handler, Array1DFloat4& output)
{
  return CreateArray1DFloat4(stream, (const std::vector<float4>&)input, array_handler, output);
}

// ------------------------------------------------------------------
// Additional Kernel Helper Functions
// ------------------------------------------------------------------
#ifdef __NVCC__

__forceinline__ __device__ bool
block_any(bool v)
{
  return __syncthreads_or(v);
}

// ------------------------------------------------------------------
// Additional Atomic Functions
// ------------------------------------------------------------------
// reference: https://github.com/treecode/Bonsai/blob/master/runtime/profiling/derived_atomic_functions.h
// https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda/51549250#51549250

__forceinline__ __device__ float
atomicMin(float* addr, float value)
{
  float old;
  old = !signbit(value) ? __int_as_float(::atomicMin((int*)addr, __float_as_int(value))) : __uint_as_float(::atomicMax((unsigned int*)addr, __float_as_uint(value)));
  return old;
}

__forceinline__ __device__ float
atomicMax(float* addr, float value)
{
  float old;
  old = !signbit(value) ? __int_as_float(::atomicMax((int*)addr, __float_as_int(value))) : __uint_as_float(::atomicMin((unsigned int*)addr, __float_as_uint(value)));
  return old;
}

#endif // __NVCC__

} // namespace vnr

#ifdef NDEBUG
#define TRACE_CUDA ((void)0)
#else
#define TRACE_CUDA CUDA_SYNC_CHECK()
#endif

#ifdef NDEBUG
#define ASSERT_THROW(X, MSG) ((void)0)
#else
#define ASSERT_THROW(X, MSG) { if (!(X)) throw std::runtime_error(MSG); }
#endif

#endif // OVR_TYPES_H
