//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //

#include "method_pathtracing.h"
#include "raytracing.h"
#include "dda.h"

#include "../types.h"
#include "../network.h"
#ifdef ENABLE_IN_SHADER
#include "../networks/tcnn_device_api.h"
#endif

#if defined(ENABLE_IN_SHADER) && defined(ENABLE_FVSRN)
#include "../networks/fvsrn_device_api.h"
#endif

#include <cuda/cuda_buffer.h>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>

#ifndef ADAPTIVE_SAMPLING
#define VARYING_MAJORANT 1
#define USE_DELTA_TRACKING_ITER 1
#endif

namespace vnr {

// NOTE: use 6 for comparison with openvkl, otherwise use 2
// #define max_num_scatters 16
#define russian_roulette_length 4

#define PHASE(albedo) (albedo * 0.6f)

struct PathTracingData : LaunchParams
{
  PathTracingData(const LaunchParams& p) : LaunchParams(p) {}

  DeviceVolume* __restrict__ volume{ nullptr };

  // belows are only useful for sampling streaming
  uint32_t* __restrict__ counter{ nullptr };
  
  uint32_t* __restrict__ pidx_and_shadow;
  vec3f* __restrict__ org{};
  vec3f* __restrict__ dir{};
  uint32_t* __restrict__ scatter_index;
  vec3f* __restrict__ sample_coord;
  float* __restrict__ sample_value;
  float* __restrict__ majorant;
  vec3f* __restrict__ L;
  vec3f* __restrict__ throughput;
#if VARYING_MAJORANT
  vec3f* __restrict__ iter_t_next{ nullptr };
  vec3i* __restrict__ iter_cell{ nullptr };
  float* __restrict__ iter_next_cell_begin{ nullptr };
#else
  float* __restrict__ iter_t{ nullptr };
#endif
  RandomTEA* __restrict__ rng;
};

static __forceinline__ __device__ uint32_t 
new_ray_index(const PathTracingData& params)
{
  return atomicAdd(params.counter, 1);
}

struct Ray
{
  float tnear{};
  float tfar{};

  uint32_t pidx{0};
  vec3f org{};
  vec3f dir{};
  bool shadow{false};
};

struct DeltaTrackingIter 
#if VARYING_MAJORANT
  : private dda::DDAIter
#endif
{
#if VARYING_MAJORANT
  using DDAIter::cell;
  using DDAIter::t_next;
  using DDAIter::next_cell_begin;
#else
  float t;
#endif
  __device__ DeltaTrackingIter() {}
  __device__ DeltaTrackingIter(const DeviceVolume& self, const Ray& ray);
  __device__ bool hashit(const DeviceVolume& self, const Ray& ray, RandomTEA& rng, float& t, float& majorant);
  __device__ bool finished(const DeviceVolume& self, const Ray& ray);
};

struct SampleStreamingPayload
{
public:
  uint32_t scatter_index = 0;
  vec3f sample_coord;
  float sample_value;
  float majorant;
  vec3f L = vec3f(0.0);
  vec3f throughput = vec3f(1.0);

  DeltaTrackingIter iter;

  RandomTEA rng;
};

inline __device__ void load(const PathTracingData& params, const uint32_t& ridx, SampleStreamingPayload& payload, Ray& ray) 
{
  const auto& self = *(params.volume);

  const uint32_t bits = params.pidx_and_shadow[ridx];
  ray.shadow = bits & 0x1;
  ray.pidx = bits >> 1;

  ray.org = params.org[ridx];              
  ray.dir = params.dir[ridx];

  ray.tnear = 0.f;
  ray.tfar = float_large;

  const bool valid = intersectVolume(ray.tnear, ray.tfar, ray.org, ray.dir, self);

  payload.scatter_index = params.scatter_index[ridx];
  payload.sample_coord  = params.sample_coord [ridx];
  payload.sample_value  = params.sample_value [ridx];
  payload.majorant      = params.majorant     [ridx];
  payload.L             = params.L            [ridx];
  payload.throughput    = params.throughput   [ridx];
  payload.rng           = params.rng          [ridx];
#if VARYING_MAJORANT
  payload.iter.t_next          = params.iter_t_next         [ridx];
  payload.iter.cell            = params.iter_cell           [ridx];
  payload.iter.next_cell_begin = params.iter_next_cell_begin[ridx];
#else
  payload.iter.t = params.iter_t[ridx];
#endif
}

inline __device__ void save(const PathTracingData& params, const uint32_t& ridx, const SampleStreamingPayload& payload, const Ray& ray)
{
  params.pidx_and_shadow[ridx] = (ray.pidx << 1) | (ray.shadow ? 0x1 : 0x0);
  params.org            [ridx] = ray.org;              
  params.dir            [ridx] = ray.dir;              
  params.scatter_index  [ridx] = payload.scatter_index;
  params.sample_coord   [ridx] = payload.sample_coord;
  params.sample_value   [ridx] = payload.sample_value;
  params.majorant       [ridx] = payload.majorant;
  params.L              [ridx] = payload.L;
  params.throughput     [ridx] = payload.throughput;
  params.rng            [ridx] = payload.rng;
#if VARYING_MAJORANT
  params.iter_t_next         [ridx] = payload.iter.t_next;
  params.iter_cell           [ridx] = payload.iter.cell;
  params.iter_next_cell_begin[ridx] = payload.iter.next_cell_begin;
#else
  params.iter_t[ridx] = payload.iter.t;
#endif
}

inline __device__ uint32_t save(const PathTracingData& params, const SampleStreamingPayload& payload, const Ray& ray)
{
  const auto ridx = new_ray_index(params);
  save(params, ridx, payload, ray);
}

/* volume decoding version */ void
do_path_tracing_trivial(cudaStream_t stream, const PathTracingData& params);

#ifdef ENABLE_IN_SHADER
/* in shader version */ void
do_path_tracing_network(cudaStream_t stream, const PathTracingData& params, NeuralVolume* network);
#endif

/* sample streaming version */ void
do_path_tracing_iterative(cudaStream_t stream, const PathTracingData& params, NeuralVolume* network, uint32_t numRays);


// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

template<typename T>
static inline T* define_buffer(char* begin, size_t& offset, size_t buffer_size)
{
  auto* ret = (T*)(begin + offset); 
  offset += buffer_size * sizeof(T);
  return ret;
}

void
MethodPathTracing::render(cudaStream_t stream, const LaunchParams& _params, StructuredRegularVolume& volume, NeuralVolume* network, bool iterative)
{
  PathTracingData params = _params;

  params.volume = (DeviceVolume*)volume.d_pointer();

  const uint32_t numPixels = params.frame.size.long_product();

  if (iterative) {
    size_t nBytes = numPixels * sizeof(SampleStreamingPayload);
    nBytes += numPixels * sizeof(uint32_t); // shadow and pidx
    nBytes += numPixels * sizeof(vec3f); // dir
    nBytes += numPixels * sizeof(vec3f); // org
    nBytes += sizeof(uint32_t); // counter

    sample_streaming_buffer.resize(nBytes, stream);
    CUDA_CHECK(cudaMemsetAsync((void*)sample_streaming_buffer.d_pointer(), 0, nBytes, stream)); // initialize all buffers

    char* begin = (char*)sample_streaming_buffer.d_pointer();
    size_t offset = 0;

    // allocate staging data
    params.pidx_and_shadow = define_buffer<uint32_t>(begin, offset, numPixels);
    params.org = define_buffer<vec3f>(begin, offset, numPixels);
    params.dir = define_buffer<vec3f>(begin, offset, numPixels);
    params.scatter_index = define_buffer<uint32_t>(begin, offset, numPixels);
    params.sample_coord = define_buffer<vec3f>(begin, offset, numPixels);
    params.sample_value = define_buffer<float>(begin, offset, numPixels);
    params.majorant = define_buffer<float>(begin, offset, numPixels);
    params.L = define_buffer<vec3f>(begin, offset, numPixels);
    params.throughput = define_buffer<vec3f>(begin, offset, numPixels);
#if VARYING_MAJORANT
    params.iter_t_next = define_buffer<vec3f>(begin, offset, numPixels);
    params.iter_cell = define_buffer<vec3i>(begin, offset, numPixels);
    params.iter_next_cell_begin = define_buffer<float>(begin, offset, numPixels);
#else
    params.iter_t = define_buffer<float>(begin, offset, numPixels);
#endif
    params.rng = define_buffer<RandomTEA>(begin, offset, numPixels);

    params.counter = define_buffer<uint32_t>(begin, offset, 1);
  }

  if (iterative) {
    do_path_tracing_iterative(stream, params, network, numPixels);
  }
  else {
#ifdef ENABLE_IN_SHADER
    if (network)
      do_path_tracing_network(stream, params, network);
    else
#endif
      do_path_tracing_trivial(stream, params);
  }
}



//------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

inline __device__ bool
delta_tracking(const DeviceVolume& self,
               const Ray& ray, // object space ray
               RandomTEA& rng,
               float& _t,
               vec3f& _albedo)
{
  const auto density_scale = self.density_scale;
  const float sigma_t = 1.f;
  const float sigma_s = 1.f;

  float t = ray.tnear;
  vec3f albedo(0);
  bool found_hit = false;

#if VARYING_MAJORANT

#if USE_DELTA_TRACKING_ITER

  float majorant;
  DeltaTrackingIter iter(self, ray);
  while (iter.hashit(self, ray, rng, t, majorant)) {
    const auto c = ray.org + t * ray.dir; // object space position
    const auto sample = sampleVolume(self.volume, c);
    const auto rgba = sampleTransferFunction(self.tfn, sample);
    if (rng.get_float() * majorant < rgba.w * density_scale) {
      albedo = rgba.xyz();
      found_hit = true;
      break;
    }
  }

#else

  float tau = -logf(1.f - rng.get_float());

  const auto dims = self.macrocell_dims;
  const vec3f m_org = ray.org * self.macrocell_spacings_rcp;
  const vec3f m_dir = ray.dir * self.macrocell_spacings_rcp;
  dda::dda3(m_org, m_dir, ray.tnear, ray.tfar, dims, false, [&](const vec3i& cell, float t0, float t1) {

    const float majorant = opacityUpperBound(self, cell) * density_scale;
    if (fabsf(majorant) <= float_epsilon) return true;

    while (t < t1) {

      const float dt = t1 - t;
      const float dtau = dt * (majorant * sigma_t);

      t = t1;
      tau -= dtau;

      if (tau > 0.f) return true;

      t = t + tau / (majorant * sigma_t); // can have division by zero error
      const auto c = ray.org + t * ray.dir; // object space position
      const auto sample = sampleVolume(self.volume, c);
      const auto rgba = sampleTransferFunction(self.tfn, sample);
      if (rng.get_float() * majorant < rgba.w * density_scale) {
        albedo = rgba.xyz();
        found_hit = true;
        return false;
      }

      tau = -logf(1.f - rng.get_float());
    }

    return true;

  });

#endif

#else

  const auto majorant = density_scale;
  while (true) {
    const vec2f xi = rng.get_floats();
    t = t + -logf(1.f - xi.x) / (majorant * sigma_t);
    if (t > ray.tfar) {
      found_hit = false;
      break;
    }
    const auto c = ray.org + t * ray.dir; // object space position
    const auto sample = sampleVolume(self.volume, c);
    const auto rgba = sampleTransferFunction(self.tfn, sample);
    if (xi.y < rgba.w * density_scale / majorant) {
      albedo = rgba.xyz();
      found_hit = true;
      break;
    }
  }

#endif

  _t = t;
  _albedo = albedo;
  return found_hit;
}

inline __device__ float luminance(const vec3f &c)
{
  return 0.212671f * c.x + 0.715160f * c.y + 0.072169f * c.z;
}

inline __device__ bool russian_roulette(vec3f& throughput, RandomTEA& rng, const int32_t& scatter_index)
{
  if (scatter_index > russian_roulette_length) { 
    float q = std::min(0.95f, /*luminance=*/reduce_max(throughput));
    if (rng.get_float() > q) {
      return true;
    }
    throughput /= q;
  }
  return false;
}

inline __device__ vec3f
path_tracing_reference(const PathTracingData& params, const affine3f& wto, RandomTEA& rng, Ray ray)
{
  const auto& self = *(params.volume);

  vec3f L = vec3f(0.0);
  vec3f throughput = vec3f(1.0);

  float t;
  vec3f albedo;

  int scatter_index = 0;
  while (intersectVolume(ray.tnear, ray.tfar, ray.org, ray.dir, self)) {

    // ray exits the volume, compute lighting
    if (!delta_tracking(self, ray, rng, t, albedo)) {
      if (scatter_index > 0) { // no light accumulation for primary rays
        L += throughput * params.light_ambient;
      }
      break;
    }

    // terminate ray
    if (russian_roulette(throughput, rng, scatter_index))
      break;
    ++scatter_index;

    // reset ray
    ray.org = ray.org + t * ray.dir;
    ray.tnear = 0.f;
    ray.tfar = float_large;
    throughput *= PHASE(albedo);

    // direct lighting
    ray.dir = xfmVector(wto, normalize(params.light_directional_dir));
    if (!delta_tracking(self, ray, rng, t, albedo)) {
      L += throughput * params.light_directional_rgb;
    }

    // scattering
    ray.dir = xfmVector(wto, uniform_sample_sphere(1.f, rng.get_floats()));
  }

  return L;
}

inline __device__ vec3f
path_tracing_traceray(const PathTracingData& params,
                      const affine3f& wto,
                      RandomTEA& rng,
                      Ray ray) // object space ray
{
  const auto& self = *(params.volume);

  vec3f L = vec3f(0.0);
  vec3f throughput = vec3f(1.0);

  float t;
  vec3f albedo;

  int scatter_index = 0;
  while (intersectVolume(ray.tnear, ray.tfar, ray.org, ray.dir, self)) {
    const bool exited = !delta_tracking(self, ray, rng, t, albedo);

    if (ray.shadow) {

      if (exited) {
        L += throughput * params.light_directional_rgb;
      }

      ray.tnear = 0.f;
      ray.tfar = float_large;
      ray.dir = xfmVector(wto, uniform_sample_sphere(1.f, rng.get_floats()));
      ray.shadow = false;
    }
    else {

      if (exited) {              // ray exits the volume, compute lighting
        if (scatter_index > 0) { // no light accumulation for primary rays
          L += throughput * params.light_ambient;
        }
        break;
      }

      if (russian_roulette(throughput, rng, scatter_index)) break;
      ++scatter_index;

      ray.org = ray.org + t * ray.dir;
      throughput *= PHASE(albedo);

      ray.tnear = 0.f;
      ray.tfar = float_large;
      ray.dir = xfmVector(wto, normalize(params.light_directional_dir));
      ray.shadow = true;
    }
  }

  return L;
}

__global__ void
path_tracing_kernel(uint32_t width, uint32_t height, const PathTracingData params)
{
  // compute pixel ID
  const size_t ix = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t iy = threadIdx.y + blockIdx.y * blockDim.y;

  if (ix >= width) return;
  if (iy >= height) return;

  assert(width == params.frame.size.x && "incorrect framebuffer size");
  assert(height == params.frame.size.y && "incorrect framebuffer size");

  const uint32_t pidx = ix + iy * width; // pixel index

  // normalized screen plane position, in [0,1]^2
  const auto& camera = params.camera;
  const vec2f screen(vec2f((float)ix + .5f, (float)iy + .5f) / vec2f(params.frame.size));

  // get the object to world transformation
  const affine3f otw = params.transform;
  const affine3f wto = otw.inverse();

  // generate ray & payload
  RandomTEA rng(params.frame_index, pidx);

  Ray ray;
  ray.tnear = 0.f;
  ray.tfar = float_large;
  ray.org = xfmPoint(wto, camera.position);
  ray.dir = xfmVector(wto, normalize(/* -z axis */ camera.direction +
                                     /* x shift */ (screen.x - 0.5f) * camera.horizontal +
                                     /* y shift */ (screen.y - 0.5f) * camera.vertical));
  ray.pidx = pidx;

  // trace ray
  const vec3f color = path_tracing_traceray(params, wto, rng, ray);

  // and write to frame buffer ... (accumilative)
  writePixelColor(params, vec4f(color, 1.f), pidx);
}

void
do_path_tracing_trivial(cudaStream_t stream, const PathTracingData& params)
{
  util::bilinear_kernel(path_tracing_kernel, 0, stream, params.frame.size.x, params.frame.size.y, params);
}



// ------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

__device__
DeltaTrackingIter::DeltaTrackingIter(const DeviceVolume& self, const Ray& ray)
{
#if VARYING_MAJORANT
  const auto& dims = self.macrocell_dims;
  const vec3f m_org = ray.org * self.macrocell_spacings_rcp;
  const vec3f m_dir = ray.dir * self.macrocell_spacings_rcp;
  DDAIter::init(m_org, m_dir, ray.tnear, ray.tfar, dims);
#else
  t = 0;
#endif
}

__device__ bool
DeltaTrackingIter::hashit(const DeviceVolume& self, const Ray& ray, RandomTEA& rng, float& rayt, float& majorant)
{
#if VARYING_MAJORANT

  const auto& dims = self.macrocell_dims;
  const vec3f m_org = ray.org * self.macrocell_spacings_rcp;
  const vec3f m_dir = ray.dir * self.macrocell_spacings_rcp;
  const auto& density_scale = self.density_scale;
  const float sigma_t = 1.f;
  const float sigma_s = 1.f;
  bool found_hit = false;
  float tau = -logf(1.f - rng.get_float());
  float t = next_cell_begin + ray.tnear;
  while (DDAIter::next(m_org, m_dir, ray.tnear, ray.tfar, dims, false, [&](const vec3i& c, float t0, float t1) {
    majorant = opacityUpperBound(self, c) * density_scale;
    if (fabsf(majorant) <= float_epsilon) return true; // move to the next macrocell
    tau -= (t1 - t) * (majorant * sigma_t);
    t = t1;
    if (tau > 0.f) return true; // move to the next macrocell  
    t = t + tau / (majorant * sigma_t); // can have division by zero error
    found_hit = true;
    next_cell_begin = t - ray.tnear;
    rayt = t;
    return false; // found a hit, terminate the loop
  })) {}
  return found_hit;

#else

  const float sigma_t = 1.f;
  const float sigma_s = 1.f;
  majorant = self.density_scale;

  t += -logf(1.f - rng.get_float()) / (majorant * sigma_t);
  rayt = ray.tnear + t;

  return (rayt <= ray.tfar);

#endif
}

__device__ bool 
DeltaTrackingIter::finished(const DeviceVolume& self, const Ray& ray)
{
#if VARYING_MAJORANT
  const auto& dims = self.macrocell_dims;
  const vec3f m_org = ray.org * self.macrocell_spacings_rcp;
  const vec3f m_dir = ray.dir * self.macrocell_spacings_rcp;
  return !DDAIter::resumable(m_dir, ray.tnear, ray.tfar, dims);
#else
  return (ray.tnear + t > ray.tfar);
#endif
}

__device__ bool
iterative_take_sample(const PathTracingData& params, 
                      SampleStreamingPayload& payload, 
                      Ray& ray)
{
  const auto& self = *((DeviceVolume*)params.volume);
  const affine3f otw = params.transform;
  const affine3f wto = otw.inverse();

  float t;
  if (payload.iter.hashit(self, ray, payload.rng, t, payload.majorant)) {
    payload.sample_coord = ray.org + t * ray.dir; // object space position  
    return true;
  }

  // ray exits the volume, compute lighting
  assert(payload.iter.finished(self, ray));
  if (payload.scatter_index > 0) { // no light accumulation for primary rays
    if (ray.shadow) {
      payload.L += payload.throughput * params.light_directional_rgb;
      
      ray.shadow = false;
      ray.dir = xfmVector(wto, uniform_sample_sphere(1.f, payload.rng.get_floats()));
      
      if (!intersectVolume(ray.tnear, ray.tfar, ray.org, ray.dir, self)) return false;
      payload.iter = DeltaTrackingIter(self, ray);

      if (payload.iter.hashit(self, ray, payload.rng, t, payload.majorant)) {
        payload.sample_coord = ray.org + t * ray.dir; // object space position  
        return true;
      }
    }
    else {
      payload.L += payload.throughput * params.light_ambient;
    }
  }
  return false;
}

__device__ bool
iterative_shade(const PathTracingData& params, 
                SampleStreamingPayload& payload,
                Ray& ray)
{
  const auto& self = *((DeviceVolume*)params.volume);
  const affine3f otw = params.transform;
  const affine3f wto = otw.inverse();

  // handle collision
  const auto rgba = sampleTransferFunction(self.tfn, payload.sample_value);
  if (payload.rng.get_float() * payload.majorant >= rgba.w * self.density_scale) return true;

  const vec3f albedo = rgba.xyz();

  if (ray.shadow) {
    ray.shadow = false;
    ray.dir = xfmVector(wto, uniform_sample_sphere(1.f, payload.rng.get_floats()));
  }
  else {
    if (russian_roulette(payload.throughput, payload.rng, payload.scatter_index)) return false;
    ++payload.scatter_index;

    ray.org = payload.sample_coord;
    ray.tnear = 0.f;
    ray.tfar = float_large;
    payload.throughput *= PHASE(albedo);

    ray.shadow = true;
    ray.dir = xfmVector(wto, normalize(params.light_directional_dir));
  }

  if (!intersectVolume(ray.tnear, ray.tfar, ray.org, ray.dir, self))
    return false;

  payload.iter = DeltaTrackingIter(self, ray);

  return true;
}

__global__ void
iterative_raygen_kernel(uint32_t numRays, const PathTracingData params) 
{
  // compute ray ID
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= numRays) return;
  const uint32_t pidx = i; // pixel index
  const uint32_t ix = pidx % params.frame.size.x;
  const uint32_t iy = pidx / params.frame.size.x;

  // generate data
  const auto& self = *((DeviceVolume*)params.volume);

  // normalized screen plane position, in [0,1]^2
  const auto& camera = params.camera;
  const vec2f screen(vec2f((float)ix + .5f, (float)iy + .5f) / vec2f(params.frame.size));

  // get the object to world transformation
  const affine3f otw = params.transform;
  const affine3f wto = otw.inverse();

  // generate ray
  Ray ray;
  ray.tnear = 0.f;
  ray.tfar = float_large;
  ray.org = xfmPoint(wto, camera.position);
  ray.dir = xfmVector(wto, normalize(/* -z axis */ camera.direction +
                                     /* x shift */ (screen.x - 0.5f) * camera.horizontal +
                                     /* y shift */ (screen.y - 0.5f) * camera.vertical));
  ray.pidx = pidx;

  // generate payload
  SampleStreamingPayload payload;
  payload.rng = RandomTEA(params.frame_index, pidx);

  float t; float majorant;
  bool alive = false;

  // initialize rays
  if (intersectVolume(ray.tnear, ray.tfar, ray.org, ray.dir, self)) {
    payload.iter = DeltaTrackingIter(self, ray);
    alive = iterative_take_sample(params, payload, ray);
  }

#if 0
  while (alive) {
    // take sample
    payload.sample_value = sampleVolume(self.volume, payload.sample_coord);

    // save
    save(params, i, payload, ray);
    load(params, i, payload, ray);

    // compute shading
    if (!iterative_shade(params, payload, ray)) break;
    if (!iterative_take_sample(params, payload, ray)) break;
  }
  writePixelColor(params, vec4f(payload.L, 1.f), ray.pidx);
#else
  if (alive) {
    save(params, payload, ray);
  }
  else {
    writePixelColor(params, vec4f(payload.L, 1.f), ray.pidx);
  }
#endif
}

__global__ void
iterative_shade_kernel(uint32_t numRays, const PathTracingData params) 
{
  // compute ray ID
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= numRays) return;
  const auto& self = *((DeviceVolume*)params.volume);

  SampleStreamingPayload payload;
  Ray ray;
  load(params, i, payload, ray);

  if (iterative_shade(params, payload, ray) && iterative_take_sample(params, payload, ray)) {
    save(params, payload, ray);
  }
  else {
    writePixelColor(params, vec4f(payload.L, 1.f), ray.pidx);
  }
}

__global__ void
iterative_sampling_groundtruth_kernel(uint32_t numRays, const PathTracingData params)
{
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= numRays) return;

  const auto& volume = params.volume->volume;
  
  const vec3f* __restrict__ inputs = params.sample_coord;
  float* __restrict__ outputs = params.sample_value;

  const auto p = inputs[i];
  outputs[i] = sampleVolume(volume, p);
}

void
iterative_sampling_batch_inference(cudaStream_t stream, uint32_t numRays, const PathTracingData& params, NeuralVolume* network)
{
  network->inference(numRays, (float*)params.sample_coord, params.sample_value, stream);
}

static bool 
iterative_ray_compaction(cudaStream_t stream, uint32_t& count, uint32_t* dptr)
{
  CUDA_CHECK(cudaMemcpyAsync(&count, dptr, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return count > 0;
}

void 
do_path_tracing_iterative(cudaStream_t stream, const PathTracingData& params, NeuralVolume* network, uint32_t numRays)
{
  CUDA_CHECK(cudaMemsetAsync(params.counter, 0, sizeof(int32_t), stream));
  util::linear_kernel(iterative_raygen_kernel, 0, stream, numRays, params);

  while (iterative_ray_compaction(stream, numRays, params.counter)) {

    if (network)
      iterative_sampling_batch_inference(stream, numRays, params, network);
    else
      util::linear_kernel(iterative_sampling_groundtruth_kernel, 0, stream, numRays, params);

    CUDA_CHECK(cudaMemsetAsync(params.counter, 0, sizeof(int32_t), stream));

    util::linear_kernel(iterative_shade_kernel, 0, stream, numRays, params);
  }
}



// ------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

#ifdef ENABLE_IN_SHADER

template<typename Volume>
inline __device__ bool
network_delta_tracking(const Volume& neuralvolume,
                       const DeviceVolume& self,
                       const Ray ray, // object space ray
                       RandomTEA& rng,
                       float& _t,
                       vec3f& _albedo,
                       bool m0)
{
  const auto density_scale = self.density_scale;
  const float sigma_t = 1.f;
  const float sigma_s = 1.f;

  float t = ray.tnear;
  vec3f albedo(0);
  bool found_hit = false;

#if VARYING_MAJORANT

#if USE_DELTA_TRACKING_ITER

  float majorant;
  DeltaTrackingIter iter = m0 ? DeltaTrackingIter(self, ray) : DeltaTrackingIter();

  bool done = !m0;
  while (true) {

    if (!done && !iter.hashit(self, ray, rng, t, majorant)) {
      done = true;
    }

    if (!block_any(!done)) break;

    assert(__syncthreads_count(1) == (blockDim.x * blockDim.y * blockDim.z));
    const auto c = ray.org + t * ray.dir; // object space position
    const auto sample = (float)neuralvolume.sample(c);
    const auto rgba = sampleTransferFunction(self.tfn, sample);

    if (!done && (rng.get_float() * majorant < rgba.w * density_scale)) {
      albedo = rgba.xyz();
      found_hit = true;
      done = true;
    }
  }

#else

  float tau = -logf(1.f - rng.get_float());

  const auto dims = self.macrocell_dims;
  const vec3f m_org = ray.org * self.macrocell_spacings_rcp;
  const vec3f m_dir = ray.dir * self.macrocell_spacings_rcp;
  dda::network_dda3(m_org, m_dir, ray.tnear, ray.tfar, dims, false, [&](const vec3i& cell, float t0, float t1, bool m_alive) {

    m_alive = m_alive && m0;

    const float majorant = m_alive ? opacityUpperBound(self, cell) * density_scale : 0.f;

    // skip empty cells
    if (!block_any(fabsf(majorant) > float_epsilon && m_alive)) return m_alive;

    // iterate within the interval
    bool m_loop = m_alive && (t < t1);
    while (block_any(m_loop)) {

      const float dt = t1 - t;
      const float dtau = dt * (majorant * sigma_t);

      if (m_loop) {
        t = t1;
        tau -= dtau;
        if (tau > 0.f) m_loop = false;
      }

      if (m_loop) t = t + tau / (majorant * sigma_t);  // can have division by zero error

      float sample;
      if (block_any(m_loop)) {
        const auto c = ray.org + t * ray.dir; // object space position
        assert(__syncthreads_count(1) == (blockDim.x * blockDim.y * blockDim.z));
        sample = (float)neuralvolume.sample(c);
      }

      if (m_loop) {
        const auto rgba = sampleTransferFunction(self.tfn, sample);
        if (rng.get_float() * majorant < rgba.w * density_scale) {
          albedo = rgba.xyz();
          found_hit = true;
          m_loop    = false;
          m_alive   = false;
        }
        tau = -logf(1.f - rng.get_float());
      }

      m_loop = m_loop && (t < t1);
    }

    return m_alive;
  });

#endif

#else

  const auto majorant = density_scale;

  bool done = !m0;
  while (block_any(!done)) {
    const vec2f xi = rng.get_floats();

    if (!done) {
      t = t + -logf(1.f - xi.x) / (majorant * sigma_t);
      if (t > ray.tfar) {
        found_hit = false;
        done = true;
      }
    }

    const auto c = ray.org + t * ray.dir; // object space position

    assert(__syncthreads_count(1) == (blockDim.x * blockDim.y * blockDim.z));
    const auto sample = (float)neuralvolume.sample(c);
    const auto rgba = sampleTransferFunction(self.tfn, sample);

    if (!done) {
      albedo = rgba.xyz();
      if (xi.y < rgba.w * density_scale / majorant) {
        found_hit = true;
        done = true;
      }
    }

  }

#endif

  _t = t;
  _albedo = albedo;

  return found_hit;
}

template<typename Volume>
inline __device__ vec3f
network_path_tracing_traceray(const Volume& neuralvolume,
                              const PathTracingData& params,
                              const affine3f& wto,
                              RandomTEA& rng,
                              Ray ray, // object space ray
                              bool m0)
{
  const auto& self = *(params.volume);

  vec3f L = vec3f(0.0);
  vec3f throughput = vec3f(1.0);

  bool m_loop = m0;

  int scatter_index = 0;
  while (true) {
    if (!intersectVolume(ray.tnear, ray.tfar, ray.org, ray.dir, self)) m_loop = false;
    if (!block_any(m_loop)) break;

    assert(__syncthreads_count(1) == (blockDim.x * blockDim.y * blockDim.z));
    float t; vec3f albedo;
    bool exited = !network_delta_tracking(neuralvolume, self, ray, rng, t, albedo, m_loop);

    if (!m_loop) continue; // no sync code blow

    if (ray.shadow) {
      if (exited) {
        L += throughput * params.light_directional_rgb;
      }
      ray.tnear = 0.f;
      ray.tfar = float_large;
      ray.dir = xfmVector(wto, uniform_sample_sphere(1.f, rng.get_floats()));
      ray.shadow = false;
    }

    else {
      if (exited) {
        if (scatter_index > 0) { // no light accumulation for primary rays
          L += throughput * params.light_ambient;
        }
        m_loop = false; continue;
      }

      if (russian_roulette(throughput, rng, scatter_index)) {
        m_loop = false; continue;
      }
      ++scatter_index;

      ray.org = ray.org + t * ray.dir;
      throughput *= PHASE(albedo);

      ray.tnear = 0.f;
      ray.tfar = float_large;
      ray.dir = xfmVector(wto, normalize(params.light_directional_dir));
      ray.shadow = true;
    }
  }

  return L;
}

template<typename Volume>
__global__ void
network_path_tracing_kernel(Volume neuralvolume, uint32_t width, uint32_t height, const PathTracingData params)
{
  neuralvolume.init();

  // compute pixel ID
  const uint32_t thread_offset = blockIdx.x * blockDim.x * blockDim.y * blockDim.z; // this can be 32bit uint
  const uint32_t thread_index = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x + thread_offset;
  const size_t ix = thread_index % width;
  const size_t iy = thread_index / width;

  bool skip = (ix >= width) || (iy >= height);
  if (!block_any(!skip)) return;

  assert(width  == params.frame.size.x && "incorrect framebuffer size");
  assert(height == params.frame.size.y && "incorrect framebuffer size");

  const uint32_t pidx = ix + iy * width; // pixel index

  // normalized screen plane position, in [0,1]^2
  const auto& camera = params.camera;
  const vec2f screen(vec2f((float)ix + .5f, (float)iy + .5f) / vec2f(params.frame.size));

  // get the object to world transformation
  const affine3f otw = params.transform;
  const affine3f wto = otw.inverse();

  // generate ray & payload
  RandomTEA rng(params.frame_index, pidx);

  Ray ray;
  ray.tnear = 0.f;
  ray.tfar = float_large;
  ray.org = xfmPoint(wto, camera.position);
  ray.dir = xfmVector(wto, normalize(/* -z axis */ camera.direction +
                                     /* x shift */ (screen.x - 0.5f) * camera.horizontal +
                                     /* y shift */ (screen.y - 0.5f) * camera.vertical));
  ray.pidx = pidx;

  // trace ray
  const vec3f color = network_path_tracing_traceray(neuralvolume, params, wto, rng, ray, !skip);

  // and write to frame buffer ... (accumilative)
  if (!skip) writePixelColor(params, vec4f(color, 1.f), pidx);
}

/* network version */
template<int N_FEATURES_PER_LEVEL>
void
do_path_tracing_network_template(cudaStream_t stream, const PathTracingData& params, NeuralVolume& network)
{
  int WIDTH = network.get_network_width();
  if (WIDTH == -1) {
    // throw std::runtime_error("incorrect MLP implementation for in-shader rendering");
    fprintf(stderr, "Incorrect MLP implementation for in-shader rendering (%s: line %d)\n", __FILE__, __LINE__);
    return;
  }

  if (WIDTH == 16) {
    TcnnDeviceVolume<16,N_FEATURES_PER_LEVEL> neuralvolume(network.get_network());
    neuralvolume.launch(network_path_tracing_kernel<TcnnDeviceVolume<16,N_FEATURES_PER_LEVEL>>, stream, params.frame.size.x, params.frame.size.y, params);
    return;
  }

  if (WIDTH == 32) {
    TcnnDeviceVolume<32,N_FEATURES_PER_LEVEL> neuralvolume(network.get_network());
    neuralvolume.launch(network_path_tracing_kernel<TcnnDeviceVolume<32,N_FEATURES_PER_LEVEL>>, stream, params.frame.size.x, params.frame.size.y, params); 
    return;
  }

  if (WIDTH == 64) {
    TcnnDeviceVolume<64,N_FEATURES_PER_LEVEL> neuralvolume(network.get_network());
    neuralvolume.launch(network_path_tracing_kernel<TcnnDeviceVolume<64,N_FEATURES_PER_LEVEL>>, stream, params.frame.size.x, params.frame.size.y, params);
    return;
  }

  fprintf(stderr, "Unsupported MLP WIDTH for in-shader rendering: %d (%s: line %d)\n", WIDTH, __FILE__, __LINE__);
}

void
do_path_tracing_network(cudaStream_t stream, const PathTracingData& params, NeuralVolume* network)
{
  int N_FEATURES_PER_LEVEL = network->get_network_features_per_level();
  if (N_FEATURES_PER_LEVEL == -1) {
    // throw std::runtime_error("incorrect encoding method for in-shader rendering");
    fprintf(stderr, "Incorrect encoding method for in-shader rendering (%s: line %d)\n", __FILE__, __LINE__);
    return;
  }

#ifdef ENABLE_FVSRN
  // special value for fV-SRN
  if (N_FEATURES_PER_LEVEL == -2) {
    const int WIDTH = network->get_network_width();
    if (WIDTH == -1) { fprintf(stderr, "Incorrect MLP implementation for in-shader rendering (%s: line %d)\n", __FILE__, __LINE__); return; }
    if (WIDTH == 16*1) { FvsrnDeviceVolume<16*1> neuralvolume(network->get_network()); neuralvolume.launch(network_path_tracing_kernel<FvsrnDeviceVolume<16*1>>, stream, params.frame.size.x, params.frame.size.y, params); return; }
    if (WIDTH == 16*2) { FvsrnDeviceVolume<16*2> neuralvolume(network->get_network()); neuralvolume.launch(network_path_tracing_kernel<FvsrnDeviceVolume<16*2>>, stream, params.frame.size.x, params.frame.size.y, params); return; }
    if (WIDTH == 16*3) { FvsrnDeviceVolume<16*3> neuralvolume(network->get_network()); neuralvolume.launch(network_path_tracing_kernel<FvsrnDeviceVolume<16*3>>, stream, params.frame.size.x, params.frame.size.y, params); return; }
    if (WIDTH == 16*4) { FvsrnDeviceVolume<16*4> neuralvolume(network->get_network()); neuralvolume.launch(network_path_tracing_kernel<FvsrnDeviceVolume<16*4>>, stream, params.frame.size.x, params.frame.size.y, params); return; }
    if (WIDTH == 16*5) { FvsrnDeviceVolume<16*5> neuralvolume(network->get_network()); neuralvolume.launch(network_path_tracing_kernel<FvsrnDeviceVolume<16*5>>, stream, params.frame.size.x, params.frame.size.y, params); return; }
    if (WIDTH == 16*6) { FvsrnDeviceVolume<16*6> neuralvolume(network->get_network()); neuralvolume.launch(network_path_tracing_kernel<FvsrnDeviceVolume<16*6>>, stream, params.frame.size.x, params.frame.size.y, params); return; }
    if (WIDTH == 16*7) { FvsrnDeviceVolume<16*7> neuralvolume(network->get_network()); neuralvolume.launch(network_path_tracing_kernel<FvsrnDeviceVolume<16*7>>, stream, params.frame.size.x, params.frame.size.y, params); return; }
    if (WIDTH == 16*8) { FvsrnDeviceVolume<16*8> neuralvolume(network->get_network()); neuralvolume.launch(network_path_tracing_kernel<FvsrnDeviceVolume<16*8>>, stream, params.frame.size.x, params.frame.size.y, params); return; }
    fprintf(stderr, "Unsupported MLP WIDTH for in-shader rendering: %d (%s: line %d)\n", WIDTH, __FILE__, __LINE__);
    return; 
  }
#endif

  if (N_FEATURES_PER_LEVEL == 1) return do_path_tracing_network_template<1>(stream, params, *network);
  if (N_FEATURES_PER_LEVEL == 2) return do_path_tracing_network_template<2>(stream, params, *network);
  if (N_FEATURES_PER_LEVEL == 4) return do_path_tracing_network_template<4>(stream, params, *network);
  if (N_FEATURES_PER_LEVEL == 8) return do_path_tracing_network_template<8>(stream, params, *network);

  fprintf(stderr, "Unsupported N_FEATURES_PER_LEVEL: %d (%s: line %d)\n", N_FEATURES_PER_LEVEL, __FILE__, __LINE__);
}

#endif // ENABLE_IN_SHADER


} // namespace vnr
