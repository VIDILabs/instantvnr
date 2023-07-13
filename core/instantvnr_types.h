#ifndef INSTANT_VNR_TYPES_H
#define INSTANT_VNR_TYPES_H

// #include <cuda.h>
#include <cuda_runtime.h>

#include <cuda/cuda_buffer.h>

#include <gdt/random/random.h>

#include "mathdef.h"
#include "array.h"

#ifdef CUDA_BUFFER_VERBOSE_MEMORY_ALLOCS
#define VNR_VERBOSE_MEMORY_ALLOCS
#endif
// #define VNR_VERBOSE_MEMORY_ALLOCS

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

#define INSTANT_VNR_NAMESPACE_BEGIN namespace vnr {
#define INSTANT_VNR_NAMESPACE_END }

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

INSTANT_VNR_NAMESPACE_BEGIN

struct MultiVolume {
public:
  struct File {
    std::string filename;
    size_t offset;
    size_t nbytes;
    bool bigendian;
    void* rawptr{ nullptr };
  };
public:
  vec3i     dims;
  ValueType type;
  range1f   range;
  vec3f scale = 0;
  vec3f translate = 0;
  std::vector<File> data;
};

struct SciVisMaterial {
public:
  const float ambient;
  const float diffuse;
  const float specular;
  const float shininess;
};

struct TransferFunction {
public:
  std::vector<vec3f> color;
  std::vector<vec2f> alpha;
  range1f range;
};

struct Camera {
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

// ------------------------------------------------------------------
// Device Definitions
// ------------------------------------------------------------------

struct DeviceTransferFunction {
  Array1DFloat4 colors;
  Array1DScalar alphas;
  range1f range;
  float range_rcp_norm;
};

struct DeviceVolume {
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

struct DeviceFrameBuffer {
    vec4f* __restrict__ rgba;
    vec2i size;
};

struct DeviceCamera {
  vec3f position;
  vec3f direction;
  vec3f horizontal;
  vec3f vertical;
};

/* shared global data */
struct LaunchParams {
  DeviceFrameBuffer frame;
  vec4f* __restrict__ accumulation{ nullptr };
  int32_t frame_index{ 0 };

  DeviceCamera camera, last_camera;

  affine3f transform;

  const float raymarching_shadow_sampling_scale = 2.f;

  /* lights */
  const float scivis_shading_scale = 0.95f;

  SciVisMaterial mat_gradient_shading{ .6f, .9f, .4f, 40.f };
  SciVisMaterial mat_full_shadow{ 1.f, .5f, .4f, 40.f };
  SciVisMaterial mat_single_shade_heuristic{ 0.8f, .2f, .4f, 40.f };

  float light_ambient{ 1.5f };
  vec3f light_directional_rgb{ 1.0f };
  vec3f light_directional_dir{ 0.7f, 0.9f, 0.4f };
};

// ------------------------------------------------------------------
// 
// ------------------------------------------------------------------

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

// ------------------------------------------------------------------
// 
// ------------------------------------------------------------------

struct TransferFunctionObject {
  DeviceTransferFunction tfn;
  cudaArray_t tfn_color_array_handler{};
  cudaArray_t tfn_alpha_array_handler{};
  void clean();
  void set_transfer_function(const std::vector<vec3f>& c, const std::vector<vec2f>& o, const range1f& r, cudaStream_t stream);
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

INSTANT_VNR_NAMESPACE_END

#endif // INSTANT_VNR_TYPES_H
