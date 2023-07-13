//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //

#include "method_raymarching.h"
#include "raytracing.h"
#include "dda.h"

#include "../network.h"
#ifdef ENABLE_IN_SHADER
#include "../networks/tcnn_device_api.h"
#endif

#if defined(ENABLE_IN_SHADER) && defined(ENABLE_FVSRN)
#include "../networks/fvsrn_device_api.h"
#endif

#include <cuda/cuda_buffer.h>

#ifndef ADAPTIVE_SAMPLING
#define ADAPTIVE_SAMPLING 1
#endif

namespace vnr {

inline int initialize_N_ITERS() 
{
  int n_iters = 16;
  if (const char* env_p = std::getenv("VNR_RM_N_ITERS")) {
    n_iters = std::stoi(env_p);
    // logging() << "[rm] N_ITERS = " << n_iters << std::endl;
  }
  return n_iters;
}

const int N_ITERS = initialize_N_ITERS();
// constexpr int N_ITERS = 16;

// constexpr int N_ITERS = 64; // fps = 6.668
// constexpr int N_ITERS = 32; // fps = 7.200
// constexpr int N_ITERS = 16; // fps = 7.249
// constexpr int N_ITERS = 8; // fps = 7.073
// constexpr int N_ITERS = 4; // fps = 6.611
// constexpr int N_ITERS = 2; // fps = 5.936
// constexpr int N_ITERS = 1; // fps = 4.786

using ShadingMode = MethodRayMarching::ShadingMode;
constexpr auto NO_SHADING             = MethodRayMarching::NO_SHADING;
constexpr auto GRADIENT_SHADING       = MethodRayMarching::GRADIENT_SHADING;
constexpr auto SINGLE_SHADE_HEURISTIC = MethodRayMarching::SINGLE_SHADE_HEURISTIC;
constexpr auto SHADOW                 = MethodRayMarching::SHADOW;

// NOTE: what is the best SoA layout here?

struct RayMarchingData : LaunchParams
{
  RayMarchingData(const LaunchParams& p) : LaunchParams(p) {}

  ShadingMode mode;

  DeviceVolume* __restrict__ volume{ nullptr };

  // belows are only useful for sampling streaming
  uint32_t* __restrict__ counter{ nullptr };

  vec3f* __restrict__ inference_input { nullptr };
  float* __restrict__ inference_output{ nullptr };

  // per ray payload (ordered by ray index)
  uint32_t* __restrict__ pixel_index{ nullptr };
  float* __restrict__ jitter{ nullptr };
  float* __restrict__ alpha{ nullptr };
  vec3f* __restrict__ color_or_org{ nullptr };
#if ADAPTIVE_SAMPLING
  vec3f* __restrict__ iter_t_next{ nullptr };
  vec3i* __restrict__ iter_cell{ nullptr };
#endif
  float* __restrict__ iter_next_cell_begin{ nullptr };

  // belows are only used by SSH
  vec3f* __restrict__ inter_highest_org  { nullptr }; // ordered by ray index
  float* __restrict__ inter_highest_alpha{ nullptr };
  vec3f* __restrict__ inter_highest_color{ nullptr };
  vec3f* __restrict__ final_highest_org  { nullptr }; // ordered by pixel index
  float* __restrict__ final_highest_alpha{ nullptr };
  vec3f* __restrict__ final_highest_color{ nullptr };
  vec4f* __restrict__ shading_color{ nullptr }; // ordered by pixel index
  float* __restrict__ jitter_ssh{ nullptr };
};

__forceinline__ __device__ uint32_t 
new_ray_index(const RayMarchingData& params)
{
  return atomicAdd(params.counter, 1);
}

struct Ray 
{
  vec3f org{};
  vec3f dir{};
  float alpha = 0.f;
  vec3f color = 0.f; // not used by shadow rays
};

struct RayMarchingIter 
#if ADAPTIVE_SAMPLING
  : private dda::DDAIter
#endif
{
#if ADAPTIVE_SAMPLING
  using DDAIter::cell;
  using DDAIter::t_next;
  using DDAIter::next_cell_begin;
#else
  float next_cell_begin{};
#endif

  __device__ RayMarchingIter() {}
  __device__ RayMarchingIter(const DeviceVolume& self, const vec3f& org, const vec3f& dir, const float tMin, const float tMax);
  bool __device__ resumable(const DeviceVolume& self, vec3f dir, float t_min, float t_max);

  template<typename F>
  __device__ void exec(const DeviceVolume& self, const vec3f& org, const vec3f& dir, const float tMin, const float tMax, const float step, const uint32_t pidx, const F& body);
};

struct SampleStreamingPayload
{
public:
  uint32_t pixel_index = 0;
  float jitter = 0.f;
  RayMarchingIter iter;

private:
  union {
    vec3f color;
    vec3f org;
  };
  float alpha = 0.f;

public:
  __device__ SampleStreamingPayload(const uint32_t pixel_index, const float jitter) : pixel_index(pixel_index), jitter(jitter), color(0) {}
  __device__ SampleStreamingPayload(const RayMarchingData& params, const uint32_t ray_index); // load a payload from memory
  __device__ void save(const RayMarchingData& params, uint32_t ridx) const;
  __device__ void as_camera_ray(const vec3f& c, const float& a) { color = c, alpha = a; }
  __device__ void as_shadow_ray(const vec3f& o) { org = o; }
  template<ShadingMode MODE> __device__ void set_ray(const Ray& ray);
  template<ShadingMode MODE> __device__ Ray compute_ray(const RayMarchingData& params) const;
};

template<> __device__ void SampleStreamingPayload::set_ray<SHADOW>(const Ray& ray) { alpha = ray.alpha; assert(ray.color == vec3f(0.f)); }
template<ShadingMode MODE> __device__ void SampleStreamingPayload::set_ray(const Ray& ray) { alpha = ray.alpha, color = ray.color; }

struct SingleShotPayload
{
  vec3f highest_org = 0.f;
  vec3f highest_color = 0.f;
  float highest_alpha = 0.f;
};

/* standard version */ void
do_raymarching_trivial(cudaStream_t stream, const RayMarchingData& params);

#ifdef ENABLE_IN_SHADER
/* network version */ void
do_raymarching_network(cudaStream_t stream, const RayMarchingData& params, NeuralVolume* network);
#endif

/* iterative version */ void
do_raymarching_iterative(cudaStream_t stream, const RayMarchingData& params, NeuralVolume* network, uint32_t numPixels);

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

template<typename T>
inline T* define_buffer(char* begin, size_t& offset, size_t buffer_size)
{
  auto* ret = (T*)(begin + offset); 
  offset += buffer_size * sizeof(T);
  return ret;
}

void
MethodRayMarching::render(cudaStream_t stream, const LaunchParams& _params, ShadingMode mode, DeviceVolume* volume, NeuralVolume* network, bool iterative)
{
  RayMarchingData params = _params;

  const uint32_t numPixels = (uint32_t)params.frame.size.long_product();

  params.volume = volume;
  params.mode = mode;

  if (iterative) {
    const uint32_t nSamplesPerCoord = (params.mode == GRADIENT_SHADING) ? 4 * N_ITERS : N_ITERS;

    size_t nBytes = numPixels * nSamplesPerCoord * sizeof(vec4f); // inference input + output
    nBytes += numPixels * sizeof(SampleStreamingPayload); // ray payloads
    nBytes += numPixels * sizeof(RayMarchingIter); // iterators
    nBytes += sizeof(uint32_t); // counter

    if (params.mode == SINGLE_SHADE_HEURISTIC)  {
      nBytes += numPixels * sizeof(SingleShotPayload) * 2;
      nBytes += numPixels * sizeof(vec4f); // shading_color
      nBytes += numPixels * sizeof(float); // jitter_ssh
    }

    sample_streaming_buffer.resize(nBytes, stream);
    CUDA_CHECK(cudaMemsetAsync((void*)sample_streaming_buffer.d_pointer(), 0, nBytes, stream)); // initialize all buffers

    char* begin = (char*)sample_streaming_buffer.d_pointer();
    size_t offset = 0;

    // allocate staging data
    params.inference_input  = define_buffer<vec3f>(begin, offset, numPixels * nSamplesPerCoord);
    params.inference_output = define_buffer<float>(begin, offset, numPixels * nSamplesPerCoord);

    // allocate payload data 
    params.alpha        = define_buffer<float>(begin, offset, numPixels);
    params.color_or_org = define_buffer<vec3f>(begin, offset, numPixels);
    params.pixel_index  = define_buffer<uint32_t>(begin, offset, numPixels);
    params.jitter       = define_buffer<float>(begin, offset, numPixels);
#if ADAPTIVE_SAMPLING
    params.iter_cell   = define_buffer<vec3i>(begin, offset, numPixels);
    params.iter_t_next = define_buffer<vec3f>(begin, offset, numPixels);
#endif
    params.iter_next_cell_begin = define_buffer<float>(begin, offset, numPixels);

    // we also need a launch index buffer
    params.counter = define_buffer<uint32_t>(begin, offset, 1);

    // single shot payloads
    if (params.mode == SINGLE_SHADE_HEURISTIC) {
      // these payloads will be compacted in every iteration
      params.jitter_ssh    = define_buffer<float>(begin, offset, numPixels);
      params.inter_highest_org   = define_buffer<vec3f>(begin, offset, numPixels);
      params.inter_highest_color = define_buffer<vec3f>(begin, offset, numPixels);
      params.inter_highest_alpha = define_buffer<float>(begin, offset, numPixels);
      // these data are fixed output
      params.final_highest_org   = define_buffer<vec3f>(begin, offset, numPixels);
      params.final_highest_color = define_buffer<vec3f>(begin, offset, numPixels);
      params.final_highest_alpha = define_buffer<float>(begin, offset, numPixels);
      params.shading_color = define_buffer<vec4f>(begin, offset, numPixels);
    }
  }

  if (iterative) {
    do_raymarching_iterative(stream, params, network, numPixels);
  }
  else {
#ifdef ENABLE_IN_SHADER
    if (network)
      do_raymarching_network(stream, params, network);
    else
#endif
      do_raymarching_trivial(stream, params);
  }
}

inline __device__ float
sample_size_scaler(const float ss, const float t0, const float t1) {
  const int32_t N = (t1-t0) / ss + 1;
  return (t1-t0) / N;
  // return ss;
}

template<typename F>
inline __device__ void
raymarching_iterator(const DeviceVolume& self, 
                     const vec3f& org, const vec3f& dir,
                     const float tMin, const float tMax, 
                     const float step, const F& body, 
                     bool debug = false)
{
#if ADAPTIVE_SAMPLING

  const auto& dims = self.macrocell_dims;
  const vec3f m_org = org * self.macrocell_spacings_rcp;
  const vec3f m_dir = dir * self.macrocell_spacings_rcp;
  dda::dda3(m_org, m_dir, tMin, tMax, dims, debug, [&](const vec3i& cell, float t0, float t1) {
    // calculate max opacity
    float r = opacityUpperBound(self, cell);
    if (fabsf(r) <= float_epsilon) return true; // the cell is empty
    // estimate a step size
    const auto ss = sample_size_scaler(adaptiveSamplingRate(step, r), t0, t1);
    // iterate within the interval
    vec2f t = vec2f(t0, min(t1, t0 + ss));
    while (t.y > t.x) {
      if (!body(t)) return false;
      t.x = t.y;
      t.y = min(t.x + ss, t1);
    }
    return true;
  });

#else

  vec2f t = vec2f(tMin, min(tMax, tMin + step));
  while ((t.y > t.x) && body(t)) {
    t.x = t.y;
    t.y = min(t.x + step, tMax);
  }

#endif
}

template<typename F>
inline __device__ void
network_raymarching_iterator(const DeviceVolume& self, 
                             const vec3f& org, const vec3f& dir,
                             const float tMin, const float tMax, 
                             const float step, const F& body, 
                             bool debug = false)
{

#if ADAPTIVE_SAMPLING

  const auto& dims = self.macrocell_dims;
  const vec3f m_org = org * self.macrocell_spacings_rcp;
  const vec3f m_dir = dir * self.macrocell_spacings_rcp;
  dda::network_dda3(m_org, m_dir, tMin, tMax, dims, debug, [&](const vec3i& cell, float t0, float t1, bool alive) 
  {
    float ss = 100.f;
    bool non_empty = true;
    if (alive) {
      // calculate max opacity
      const float r = opacityUpperBound(self, cell);
      if (fabsf(r) <= float_epsilon) non_empty = false; // the cell is empty
      // estimate a step size
      ss = adaptiveSamplingRate(step, r);
    }
    if (!block_any(non_empty && alive)) return alive;
    // iterate within the interval
    vec2f t = vec2f(t0, min(t1, t0 + ss));
    while (block_any(alive && (t.y > t.x))) {
      alive = body(t, alive);
      t.x = t.y;
      t.y = min(t.x + ss, t1);
    }
    return alive;
  });

#else

  vec2f t = vec2f(tMin, min(tMax, tMin + step));
  bool alive = (t.y > t.x);
  while (block_any(alive && (t.y > t.x))) {
    alive = body(t, alive);
    t.x = t.y;
    t.y = min(t.x + step, tMax);
  }

#endif

}

//------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

inline __device__ float
raymarching_transmittance(const DeviceVolume& self,
                          const RayMarchingData& params,
                          const vec3f& org, const vec3f& dir,
                          float t0, float t1,
                          float sampling_scale,
                          RandomTEA& rng)
{
  const auto marching_step = sampling_scale * self.step;

  float alpha(0);

  if (intersectVolume(t0, t1, org, dir, self)) {
    // jitter ray to remove ringing effects
    const float jitter = rng.get_floats().x;

    // start marching
    raymarching_iterator(self, org, dir, t0, t1, marching_step, [&](const vec2f& t) {
      // sample data value
      const auto p = org + lerp(jitter, t.x, t.y) * dir; // object space position
      const auto sampleValue = sampleVolume(self.volume, p);

      // classification
      vec3f sampleColor;
      float sampleAlpha;
      sampleTransferFunction(self.tfn, sampleValue, sampleColor, sampleAlpha);
      opacityCorrection(self, t.y - t.x, sampleAlpha);

      // blending
      alpha += (1.f - alpha) * sampleAlpha;
      return alpha < nearly_one;
    });
  }

  return 1.f - alpha;
}

inline __device__ vec4f
raymarching_traceray(const DeviceVolume& self,
                     const RayMarchingData& params,
                     const affine3f& wto, // world to object
                     const affine3f& otw, // object to world
                     const Ray& ray, float t0, float t1,
                     RandomTEA& rng)
{
  const auto& marchingStep = self.step;
  const auto& gradientStep = self.grad_step;
  const auto& shadingScale = params.scivis_shading_scale;

  // vec3f gradient = 0.f;
  vec3f highestOrg   = 0.f;
  vec3f highestColor = 0.f;
  float highestAlpha = 0.f;

  float alpha(0);
  vec3f color(0);

  if (intersectVolume(t0, t1, ray.org, ray.dir, self)) {
    // jitter ray to remove ringing effects
    const float jitter = rng.get_floats().x;

    // start marching
    raymarching_iterator(self, ray.org, ray.dir, t0, t1, marchingStep, [&](const vec2f& t) {
      assert(t.x < t.y);

      // sample data value
      const auto p = ray.org + lerp(jitter, t.x, t.y) * ray.dir; // object space position
      const auto sampleValue = sampleVolume(self.volume, p);

      // classification
      vec3f sampleColor;
      float sampleAlpha;
      sampleTransferFunction(self.tfn, sampleValue, sampleColor, sampleAlpha);
      opacityCorrection(self, t.y - t.x, sampleAlpha);

      // access gradient
      const vec3f No = -sampleGradient(self.volume, p, sampleValue, gradientStep); // sample gradient
      const vec3f Nw = xfmNormal(otw, No);

      const float tr = 1.f - alpha;

      // compute shading
      if (params.mode == GRADIENT_SHADING) {
        const auto dir = xfmVector(otw, ray.dir);
        const vec3f shadingColor = shade_scivis_light(dir, Nw, sampleColor, 
                                                      params.mat_gradient_shading, 
                                                      params.light_ambient, 
                                                      params.light_directional_rgb, 
                                                      params.light_directional_dir);
        sampleColor = lerp(shadingScale, sampleColor, shadingColor);
      }
      else if (params.mode == SINGLE_SHADE_HEURISTIC) {
        // remember point of highest density for deferred shading
        if (highestAlpha < (1.f - alpha) * sampleAlpha) {
          highestOrg = p; // object space
          highestColor = sampleColor;
          highestAlpha = (1.f - alpha) * sampleAlpha;
        }
        // gradient += tr * Nw; // accumulate gradient for SSH
      }

      color += tr * sampleColor * sampleAlpha;
      alpha += tr * sampleAlpha;

      return alpha < nearly_one;
    });

    if (highestAlpha > 0.f) {
      // object space to world space
      const auto ldir = xfmVector(wto, normalize(params.light_directional_dir));
      const auto rdir = xfmVector(otw, ray.dir);
      // single shade
      const float transmittance = raymarching_transmittance(self, params, highestOrg, ldir, 0.f, float_large, 
                                                            params.raymarching_shadow_sampling_scale, rng);
      // const vec3f shadingColor = shade_scivis_light(rdir, gradient, highestColor, 
      //                                               params.mat_single_shade_heuristic, 
      //                                               params.light_ambient, 
      //                                               params.light_directional_rgb, 
      //                                               params.light_directional_dir);
      color = lerp(shadingScale, color, highestColor * alpha * transmittance);
    }
  }

  return vec4f(color, alpha);
}

__global__ void
raymarching_kernel(uint32_t width, uint32_t height, const RayMarchingData params)
{
  // compute pixel ID
  const size_t ix = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t iy = threadIdx.y + blockIdx.y * blockDim.y;

  if (ix >= width)  return;
  if (iy >= height) return;

  const auto& volume = *params.volume;
  assert(width  == params.frame.size.x && "incorrect framebuffer size");
  assert(height == params.frame.size.y && "incorrect framebuffer size");

  // normalized screen plane position, in [0,1]^2
  const auto& camera = params.camera;
  const vec2f screen(vec2f((float)ix + .5f, (float)iy + .5f) / vec2f(params.frame.size));

  // get the object to world transformation
  const affine3f otw = params.transform;
  const affine3f wto = otw.inverse();

  // pixel index
  const uint32_t fbIndex = ix + iy * width;

  // random number generator
  RandomTEA rng_state(params.frame_index, fbIndex);

  // generate ray direction
  Ray ray;
  ray.org = xfmPoint(wto, camera.position);
  ray.dir = xfmVector(wto, normalize(/* -z axis */ camera.direction +
                                     /* x shift */ (screen.x - 0.5f) * camera.horizontal +
                                     /* y shift */ (screen.y - 0.5f) * camera.vertical));

  // trace ray
  const vec4f output = raymarching_traceray(volume, params, wto, otw, ray, 0.f, float_large, rng_state);

  // and write to frame buffer ...
  writePixelColor(params, output, fbIndex);
}

void
do_raymarching_trivial(cudaStream_t stream, const RayMarchingData& params)
{
  util::bilinear_kernel(raymarching_kernel, 0, stream, params.frame.size.x, params.frame.size.y, params);
}



// ------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

__device__
RayMarchingIter::RayMarchingIter(const DeviceVolume& self, const vec3f& org, const vec3f& dir, const float tMin, const float tMax)
{
#if ADAPTIVE_SAMPLING
  const auto& dims = self.macrocell_dims;
  const vec3f m_org = org * self.macrocell_spacings_rcp;
  const vec3f m_dir = dir * self.macrocell_spacings_rcp;
  DDAIter::init(m_org, m_dir, tMin, tMax, dims);
#endif
}

template<typename F>
__device__ void
RayMarchingIter::exec(const DeviceVolume& self, const vec3f& org, const vec3f& dir, const float tMin, const float tMax, const float step, const uint32_t pidx, const F& body)
{
#if ADAPTIVE_SAMPLING

  const auto& dims = self.macrocell_dims;
  const vec3f m_org = org * self.macrocell_spacings_rcp;
  const vec3f m_dir = dir * self.macrocell_spacings_rcp;

  const auto lambda = [&](const vec3i& cell, float t0, float t1) {
    // calculate max opacity
    float r = opacityUpperBound(self, cell);
    if (fabsf(r) <= float_epsilon) return true; // the cell is empty
    // estimate a step size
    const auto ss = adaptiveSamplingRate(step, r);
    // iterate within the interval
    vec2f t = vec2f(t0, min(t1, t0 + ss));
    while (t.y > t.x) {
      DDAIter::next_cell_begin = t.y - tMin;
      if (!body(t)) return false;
      t.x = t.y;
      t.y = min(t.x + ss, t1);
    }
    return true;
  };

  while (DDAIter::next(m_org, m_dir, tMin, tMax, dims, false, lambda)) {}

#else

  vec2f t;
  t.x = max(tMin + next_cell_begin, tMin);
  t.y = min(t.x + step, tMax);
  while (t.y > t.x) {
    next_cell_begin = t.y - tMin;
    if (!body(t)) return;
    t.x = t.y;
    t.y = min(t.x + step, tMax);
  }

  next_cell_begin = float_large;
  return;

#endif
}

bool __device__
RayMarchingIter::resumable(const DeviceVolume& self, vec3f dir, float tMin, float tMax)
{
#if ADAPTIVE_SAMPLING
  const auto& dims = self.macrocell_dims;
  const vec3f m_dir = dir * self.macrocell_spacings_rcp;
  return DDAIter::resumable(m_dir, tMin, tMax, dims);
#else
  return tMin + next_cell_begin < tMax;
#endif
}

__device__
SampleStreamingPayload::SampleStreamingPayload(const RayMarchingData& params, const uint32_t ray_index) 
{
  pixel_index = params.pixel_index[ray_index];
  jitter = params.jitter[ray_index];
  alpha = params.alpha[ray_index];
  color = params.color_or_org[ray_index];
#if ADAPTIVE_SAMPLING
  iter.cell = params.iter_cell[ray_index];
  iter.t_next = params.iter_t_next[ray_index];
#endif
  iter.next_cell_begin = params.iter_next_cell_begin[ray_index];
}

__device__ void
SampleStreamingPayload::save(const RayMarchingData& params, uint32_t ridx) const
{
  params.pixel_index[ridx] = pixel_index;
  params.jitter[ridx] = jitter;
  params.alpha[ridx] = alpha;
  params.color_or_org[ridx] = color;
#if ADAPTIVE_SAMPLING
  params.iter_cell[ridx] = iter.cell;
  params.iter_t_next[ridx] = iter.t_next;
#endif
  params.iter_next_cell_begin[ridx] = iter.next_cell_begin;
}

template<> 
__device__ Ray 
SampleStreamingPayload::compute_ray<SHADOW>(const RayMarchingData& params) const
{
  // get the object to world transformation
  const affine3f& otw = params.transform;
  const affine3f wto = otw.inverse();
  // generate ray direction
  Ray ray;
  ray.org = org;
  ray.dir = xfmVector(wto, normalize(params.light_directional_dir));
  ray.alpha = alpha;
  ray.color = 0.f; // not used by shadow rays
  return ray;
}

template<ShadingMode MODE> 
__device__ Ray 
SampleStreamingPayload::compute_ray(const RayMarchingData& params) const
{
  const auto& fbIndex = pixel_index;

  // compute pixel ID
  const uint32_t ix = fbIndex % params.frame.size.x;
  const uint32_t iy = fbIndex / params.frame.size.x;

  // normalized screen plane position, in [0,1]^2
  const auto& camera = params.camera;
  const vec2f screen(vec2f((float)ix + .5f, (float)iy + .5f) / vec2f(params.frame.size));

  // get the object to world transformation
  const affine3f& otw = params.transform;
  const affine3f wto = otw.inverse();

  // generate ray direction
  Ray ray;
  ray.org = xfmPoint(wto, camera.position);
  ray.dir = xfmVector(wto, normalize(/* -z axis */ camera.direction +
                                     /* x shift */ (screen.x - 0.5f) * camera.horizontal +
                                     /* y shift */ (screen.y - 0.5f) * camera.vertical));
  ray.alpha = alpha;
  ray.color = color;
  return ray;
}

template<ShadingMode MODE> 
__global__ void
iterative_intersect_kernel(uint32_t numRays, const RayMarchingData params, int N_ITERS)
{
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= numRays) return;

  // other constants
  const auto& self = *(params.volume);
  const auto& gradientStep = self.grad_step;
  const auto& volume = self.volume;

  // load payloads and rays
  SampleStreamingPayload payload(params, i);
  const Ray ray = payload.compute_ray<MODE>(params);

  float tmin = 0.f, tmax = float_large;
  const bool hashit = intersectVolume(tmin, tmax, ray.org, ray.dir, self);
  assert(hashit);

  vec3f* __restrict__ coords = (vec3f*)params.inference_input;

  int k = 0;
  payload.iter.exec(self, ray.org, ray.dir, tmin, tmax, self.step, payload.pixel_index, [&](const vec2f& t) {
    assert(k < N_ITERS);
    assert(t.x < t.y);

    // object space position
    const vec3f c = ray.org + lerp(payload.jitter, t.x, t.y) * ray.dir;
    coords[numRays * k + i] = c;

    // object space gradient
    if (MODE == GRADIENT_SHADING) {
      const vec3f gx = c + vec3f(gradientStep.x, 0, 0);
      const vec3f gy = c + vec3f(0, gradientStep.y, 0);
      const vec3f gz = c + vec3f(0, 0, gradientStep.z);
      coords[1 * numRays * N_ITERS + (numRays * k + i)] = gx; 
      coords[2 * numRays * N_ITERS + (numRays * k + i)] = gy; 
      coords[3 * numRays * N_ITERS + (numRays * k + i)] = gz; 
    }

    return (++k) < N_ITERS;
  });
}

template<ShadingMode MODE> 
__global__ void
iterative_compose_kernel(uint32_t numRays, const RayMarchingData params, int N_ITERS)
{
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= numRays) return;

  const auto& self = *(params.volume);
  const auto& otw = params.transform;
  const auto& gradientStep = self.grad_step;
  const auto& shadingScale = params.scivis_shading_scale;
  const auto* __restrict__ samples = params.inference_output;

  SampleStreamingPayload payload(params, i);
  Ray ray = payload.compute_ray<MODE>(params);

  SingleShotPayload ssh;
  if (MODE == SINGLE_SHADE_HEURISTIC) {
    ssh.highest_org   = params.inter_highest_org[i];
    ssh.highest_color = params.inter_highest_color[i];
    ssh.highest_alpha = params.inter_highest_alpha[i];
  }

  float tmin = 0.f, tmax = float_large;
  const bool hashit = intersectVolume(tmin, tmax, ray.org, ray.dir, self);
  assert(hashit);

  int k = 0;
  payload.iter.exec(self, ray.org, ray.dir, tmin, tmax, self.step, payload.pixel_index, [&](const vec2f& t) {
    assert(k < N_ITERS);
    assert(t.x < t.y);

    // classification
    const auto c = ray.org + lerp(payload.jitter, t.x, t.y) * ray.dir;
    const auto sampleValue = samples[numRays * k + i];
    vec3f sampleColor;
    float sampleAlpha;
    sampleTransferFunction(self.tfn, sampleValue, sampleColor, sampleAlpha);
    opacityCorrection(self, t.y - t.x, sampleAlpha);

    // shading
    if (MODE == GRADIENT_SHADING) {
      // compute sample gradient
      const auto fgx = samples[1 * numRays * N_ITERS + numRays * k + i];
      const auto fgy = samples[2 * numRays * N_ITERS + numRays * k + i];
      const auto fgz = samples[3 * numRays * N_ITERS + numRays * k + i];
      const vec3f No = -vec3f(fgx - sampleValue, fgy - sampleValue, fgz - sampleValue) / gradientStep;
      const vec3f Nw = xfmNormal(otw, No);
      // calculate lighting in the world space 
      const auto dir = xfmVector(otw, ray.dir);
      const vec3f shadingColor = shade_scivis_light(dir, Nw, sampleColor, 
                                                    params.mat_gradient_shading, 
                                                    params.light_ambient, 
                                                    params.light_directional_rgb, 
                                                    params.light_directional_dir);
      sampleColor = lerp(shadingScale, sampleColor, shadingColor);
    }
    else if (MODE == SINGLE_SHADE_HEURISTIC) {
      if (ssh.highest_alpha < (1.f - ray.alpha) * sampleAlpha) {
        ssh.highest_org = c;
        ssh.highest_color = sampleColor;
        ssh.highest_alpha = (1.f - ray.alpha) * sampleAlpha;
      }
    }
    
    // blending
    const float tr = 1.f - ray.alpha;
    ray.alpha += tr * sampleAlpha;
    if (MODE != SHADOW) {
      ray.color += tr * sampleColor * sampleAlpha;
    }

    // conditions to continue iterating
    return ((++k) < N_ITERS) && (ray.alpha < nearly_one);
  });

  payload.set_ray<MODE>(ray);
  const bool resumable = payload.iter.resumable(self, ray.dir, tmin, tmax);
  if (ray.alpha < nearly_one && resumable) {
    const auto ridx = new_ray_index(params);
    payload.save(params, ridx);
    if (MODE == SINGLE_SHADE_HEURISTIC) {
      params.inter_highest_org[ridx]   = ssh.highest_org;
      params.inter_highest_color[ridx] = ssh.highest_color;
      params.inter_highest_alpha[ridx] = ssh.highest_alpha;
    }
  }
  else {
    if (MODE == SHADOW) {
      const uint32_t& pidx = payload.pixel_index;
      const float transmittance = 1.f - ray.alpha;
      vec4f shadingColor = params.shading_color[pidx];
      shadingColor.xyz() = lerp(shadingScale, shadingColor.xyz(), params.final_highest_color[pidx] * shadingColor.w * transmittance);
      writePixelColor(params, shadingColor, pidx);
    }
    else if (MODE == SINGLE_SHADE_HEURISTIC) {
      const uint32_t& pidx = payload.pixel_index;
      params.final_highest_org[pidx] = ssh.highest_org;
      params.final_highest_color[pidx] = ssh.highest_color;
      params.final_highest_alpha[pidx] = ssh.highest_alpha;
      params.shading_color[pidx] = vec4f(ray.color, ray.alpha);
    }
    else {
      writePixelColor(params, vec4f(ray.color, ray.alpha), payload.pixel_index);
    }
  }
}

__global__ void
iterative_raygen_kernel_camera(uint32_t numRays, const RayMarchingData params) 
{
  // compute ray ID
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= numRays) return;

  // generate data
  const auto& self = *((DeviceVolume*)params.volume);

  // random number generator
  RandomTEA rng = RandomTEA(params.frame_index, i);
  vec2f jitters = rng.get_floats();

  // payload & ray
  SampleStreamingPayload payload(i, jitters.x);
  const Ray ray = payload.compute_ray<NO_SHADING>(params);

  // intersect with volume bbox & write outputs
  float tmin = 0.f, tmax = float_large;
  if (intersectVolume(tmin, tmax, ray.org, ray.dir, self)) {
    payload.iter = RayMarchingIter(self, ray.org, ray.dir, tmin, tmax);

    const auto ridx = new_ray_index(params);
    payload.save(params, ridx);

    if (params.mode == SINGLE_SHADE_HEURISTIC) {
      params.jitter_ssh[payload.pixel_index] = jitters.y;
    }
  }
  else {
    if (params.mode != SINGLE_SHADE_HEURISTIC) {
      writePixelColor(params, vec4f(ray.color, ray.alpha), payload.pixel_index);
    }
  }
}

__global__ void
iterative_raygen_kernel_shadow(uint32_t numRays, const RayMarchingData params) 
{
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= numRays) return;

  const auto& self = *((DeviceVolume*)params.volume);

  SampleStreamingPayload payload(i, params.jitter_ssh[i]);
  payload.as_shadow_ray(params.final_highest_org[i]);
  
  const Ray ray = payload.compute_ray<SHADOW>(params);

  float tmin = 0.f, tmax = float_large;
  if (intersectVolume(tmin, tmax, ray.org, ray.dir, self) && (params.final_highest_alpha[i] > 0.f)) {
    payload.iter = RayMarchingIter(self, ray.org, ray.dir, tmin, tmax);

    const auto ridx = new_ray_index(params);
    payload.save(params, ridx); 
  }
  else {
    writePixelColor(params, params.shading_color[i], i);
  }
}

__global__ void
iterative_sampling_groundtruth_kernel(uint32_t numRays, const RayMarchingData params)
{
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= numRays) return;

  const auto& volume = params.volume->volume;
  
  const vec3f* __restrict__ inputs = params.inference_input;
  float* __restrict__ outputs = params.inference_output;

  const auto p = inputs[i];
  outputs[i] = sampleVolume(volume, p);
}

void
iterative_sampling_batch_inference(cudaStream_t stream, uint32_t numRays, const RayMarchingData& params, NeuralVolume* network)
{
  network->inference(numRays, (float*)params.inference_input, params.inference_output, stream);
}

inline bool 
iterative_ray_compaction(cudaStream_t stream, uint32_t& count, uint32_t* dptr)
{
  CUDA_CHECK(cudaMemcpyAsync(&count, dptr, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return count > 0;
}

template<ShadingMode MODE> 
void iterative_raymarching_loop(cudaStream_t stream, const RayMarchingData& params, NeuralVolume* network, uint32_t numRays)
{
  const uint32_t numCoordsPerSample = (MODE == GRADIENT_SHADING) ? 4 * N_ITERS : N_ITERS;

  CUDA_CHECK(cudaMemsetAsync(params.counter, 0, sizeof(int32_t), stream));
  if (MODE == SHADOW) {
    util::linear_kernel(iterative_raygen_kernel_shadow, 0, stream, numRays, params);
  }
  else {
    util::linear_kernel(iterative_raygen_kernel_camera, 0, stream, numRays, params);
  }

  while (iterative_ray_compaction(stream, numRays, params.counter)) {
    // Actually, we could have merged the intersection step with raygen and compose. However, there was a wired error 
    // and I did not figure out irs origin. Also, having the intersection step inside raygen and compose did not bring
    // obvious performance benefit, so I left it as it is for now.
    util::linear_kernel(iterative_intersect_kernel<MODE>, 0, stream, numRays, params, N_ITERS);

    if (network)
      iterative_sampling_batch_inference(stream, numCoordsPerSample * numRays, params, network);
    else
      util::linear_kernel(iterative_sampling_groundtruth_kernel, 0, stream, numCoordsPerSample * numRays, params);

    CUDA_CHECK(cudaMemsetAsync(params.counter, 0, sizeof(int32_t), stream));
    util::linear_kernel(iterative_compose_kernel<MODE>, 0, stream, numRays, params, N_ITERS);
  }
}

void
do_raymarching_iterative(cudaStream_t stream, const RayMarchingData& params, NeuralVolume* network, uint32_t numRays)
{
  if (params.mode == NO_SHADING) {
    iterative_raymarching_loop<NO_SHADING>(stream, params, network, numRays);
  }
  else if (params.mode == GRADIENT_SHADING) {
    iterative_raymarching_loop<GRADIENT_SHADING>(stream, params, network, numRays);
  }
  else if (params.mode == SINGLE_SHADE_HEURISTIC) {
    iterative_raymarching_loop<SINGLE_SHADE_HEURISTIC>(stream, params, network, numRays);
    iterative_raymarching_loop<SHADOW>(stream, params, network, numRays);
  }
}



//------------------------------------------------------------------------------
//
// ------------------------------------------------------------------------------

#ifdef ENABLE_IN_SHADER

template<typename Volume>
__device__ float
network_raymarching_transmittance(const Volume& neuralvolume,
                                  const DeviceVolume& self,
                                  const RayMarchingData& params,
                                  const vec3f& org, const vec3f& dir,
                                  float t0, float t1,
                                  float sampling_scale,
                                  RandomTEA& rng)
{
  const auto marching_step = sampling_scale * self.step;

  const auto sampler = [&] __device__ (vec3f p) {
    return (float)neuralvolume.sample(p);
    // return sampleVolume(self.volume, p);
  };

  float alpha(0);

  const auto jitter = rng.get_floats().x;

  // intersect the ray with volume bbox
  if (!block_any(intersectVolume(t0, t1, org, dir, self))) {
    return 1.f - alpha;
  }

  // start marching
  network_raymarching_iterator(self, org, dir, t0, t1, marching_step, [&](const vec2f& t, bool alive)
  {
    // sample neural volume
    const vec3f p = org + lerp(jitter, t.x, t.y) * dir; // object space position
    const float value = sampler(p);

    if (alive && (t.y > t.x)) {
      // classification
      vec3f sampleColor;
      float sampleAlpha;
      sampleTransferFunction(self.tfn, value, sampleColor, sampleAlpha);
      opacityCorrection(self, t.y - t.x, sampleAlpha);

      // blending
      alpha += (1.f - alpha) * sampleAlpha;
    }

    return alive && (alpha < nearly_one);
  });

  return 1.f - alpha;
}

template<typename Volume>
__device__ vec4f
network_raymarching_traceray(const Volume& neuralvolume,
                             const DeviceVolume& self,
                             const RayMarchingData& params,
                             const affine3f& wto, /* world to object */
                             const affine3f& otw, /* world to object */
                             const Ray& ray, float t0, float t1,
                             RandomTEA& rng)
{
  const auto sampler = [&] __device__ (vec3f p) {
    return (float)neuralvolume.sample(p);
    // return sampleVolume(self.volume, p);
  };

  const auto& gradientStep = self.grad_step;
  const auto& shadingScale = params.scivis_shading_scale;

  // vec3f averageGradient = 0.f;
  vec3f highestOrg = 0.f;
  vec3f highestColor = 0.f;
  float highestAlpha = 0.f;

  float alpha(0);
  vec3f color(0);

  const auto jitter = rng.get_floats().x;

  // intersect the ray with volume bbox
  if (!block_any(intersectVolume(t0, t1, ray.org, ray.dir, self))) {
    return vec4f(color, alpha);
  }

  // start marching
  network_raymarching_iterator(self, ray.org, ray.dir, t0, t1, self.step, [&](const vec2f& t, bool alive)
  {
    // sample data value
    const vec3f p = ray.org + lerp(jitter, t.x, t.y) * ray.dir; // object space position
    const float value = sampler(p);
    const vec3f gradient = (params.mode == GRADIENT_SHADING) 
      ? -sampleGradient(p, value, gradientStep, sampler) : vec3f(0);

    if (alive && (t.y > t.x)) {
      // classification
      vec3f sampleColor;
      float sampleAlpha;
      sampleTransferFunction(self.tfn, value, sampleColor, sampleAlpha);
      opacityCorrection(self, t.y - t.x, sampleAlpha);

      // shading
      const float tr = 1.f - alpha;
      if (params.mode == GRADIENT_SHADING) {
        const vec3f Nw = xfmNormal(otw, gradient);
        const vec3f shadingColor = shade_scivis_light(xfmVector(otw, ray.dir), Nw, sampleColor, 
                                                      params.mat_gradient_shading, 
                                                      params.light_ambient, 
                                                      params.light_directional_rgb, 
                                                      params.light_directional_dir);
        sampleColor = lerp(shadingScale, sampleColor, shadingColor);
      }
      else if (params.mode == SINGLE_SHADE_HEURISTIC) {
        // remember point of highest density for deferred shading
        if (highestAlpha < (1.f - alpha) * sampleAlpha) {
          highestOrg = p; // object space
          highestColor = sampleColor;
          highestAlpha = (1.f - alpha) * sampleAlpha;
        }
        // const vec3f Nw = xfmNormal(otw, gradient);
        // averageGradient += tr * Nw; // accumulate gradient for SSH
      }

      // blending
      color += tr * sampleColor * sampleAlpha;
      alpha += tr * sampleAlpha;
    }

    return alive && (alpha < nearly_one);
  });

  // single shade
  if (block_any(highestAlpha > 0.f)) {
    // calculate transmittance
    const float transmittance = network_raymarching_transmittance(neuralvolume, self, params, highestOrg, 
                                                                  xfmVector(wto, normalize(params.light_directional_dir)), 
                                                                  0.f, float_large, params.raymarching_shadow_sampling_scale, rng);
    // shading
    if (highestAlpha > 0.f) {
      // const vec3f shadingColor = shade_scivis_light(xfmVector(otw, ray.dir), averageGradient, highestColor, 
      //                                               params.mat_single_shade_heuristic, 
      //                                               params.light_ambient, 
      //                                               params.light_directional_rgb, 
      //                                               params.light_directional_dir);
      color = lerp(shadingScale, color, highestColor * alpha * transmittance);
    }
  }

  return vec4f(color, alpha);
}

template<typename Volume>
__global__ void
network_raymarching_kernel(Volume neuralvolume, uint32_t width, uint32_t height, const RayMarchingData params)
{
  neuralvolume.init();

  const uint32_t off = blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
  const uint32_t i = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x + off;
  const uint32_t ix = i % width; // compute pixel ID
  const uint32_t iy = i / width;

  const bool skip = (ix >= width) || (iy >= height);
  if (!block_any(!skip)) return;

  const auto& volume = *(params.volume);
  assert(width  == params.frame.size.x && "incorrect framebuffer size");
  assert(height == params.frame.size.y && "incorrect framebuffer size");

  // normalized screen plane position, in [0,1]^2
  const auto& camera = params.camera;
  const vec2f screen(vec2f((float)ix + .5f, (float)iy + .5f) / vec2f(params.frame.size));

  // get the object to world transformation
  affine3f otw = params.transform;
  affine3f wto = otw.inverse();

  // pixel index
  const uint32_t fbIndex = ix + iy * width;

  // random number generator
  RandomTEA rng(params.frame_index, fbIndex);

  // generate ray direction
  Ray ray;
  ray.org = xfmPoint(wto, camera.position);
  ray.dir = xfmVector(wto, normalize(/* -z axis */ camera.direction +
                                     /* x shift */ (screen.x - 0.5f) * camera.horizontal +
                                     /* y shift */ (screen.y - 0.5f) * camera.vertical));

  // trace ray
  const vec4f output = network_raymarching_traceray(neuralvolume, volume, params, wto, otw, ray, 0.f, float_large, rng);

  // and write to frame buffer ...
  if (!skip) {
    writePixelColor(params, output, fbIndex);
  }
}

/* network version */
template<int N_FEATURES_PER_LEVEL>
void
do_raymarching_network_template(cudaStream_t stream, const RayMarchingData& params, NeuralVolume& network)
{
  int WIDTH = network.get_network_width();
  if (WIDTH == -1) {
    // throw std::runtime_error("incorrect MLP implementation for in-shader rendering");
    fprintf(stderr, "Incorrect MLP implementation for in-shader rendering (%s: line %d)\n", __FILE__, __LINE__);
    return;
  }

  if (WIDTH == 16) {
    TcnnDeviceVolume<16,N_FEATURES_PER_LEVEL> neuralvolume(network.get_network());
    neuralvolume.launch2D(network_raymarching_kernel<TcnnDeviceVolume<16,N_FEATURES_PER_LEVEL>>, stream, params.frame.size.x, params.frame.size.y, params);
    return;
  }

  if (WIDTH == 32) {
    TcnnDeviceVolume<32,N_FEATURES_PER_LEVEL> neuralvolume(network.get_network());
    neuralvolume.launch2D(network_raymarching_kernel<TcnnDeviceVolume<32,N_FEATURES_PER_LEVEL>>, stream, params.frame.size.x, params.frame.size.y, params); 
    return;
  }

  if (WIDTH == 64) {
    TcnnDeviceVolume<64,N_FEATURES_PER_LEVEL> neuralvolume(network.get_network());
    neuralvolume.launch2D(network_raymarching_kernel<TcnnDeviceVolume<64,N_FEATURES_PER_LEVEL>>, stream, params.frame.size.x, params.frame.size.y, params);
    return;
  }

  fprintf(stderr, "Unsupported MLP WIDTH for in-shader rendering: %d (%s: line %d)\n", WIDTH, __FILE__, __LINE__);
}

void
do_raymarching_network(cudaStream_t stream, const RayMarchingData& params, NeuralVolume* network)
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
    if (WIDTH == -1)  { fprintf(stderr, "Incorrect MLP implementation for in-shader rendering (%s: line %d)\n", __FILE__, __LINE__); return; }
    if (WIDTH == 16*1) { FvsrnDeviceVolume<16*1> neuralvolume(network->get_network()); neuralvolume.launch(network_raymarching_kernel<FvsrnDeviceVolume<16*1>>, stream, params.frame.size.x, params.frame.size.y, params); return; }
    if (WIDTH == 16*2) { FvsrnDeviceVolume<16*2> neuralvolume(network->get_network()); neuralvolume.launch(network_raymarching_kernel<FvsrnDeviceVolume<16*2>>, stream, params.frame.size.x, params.frame.size.y, params); return; }
    if (WIDTH == 16*3) { FvsrnDeviceVolume<16*3> neuralvolume(network->get_network()); neuralvolume.launch(network_raymarching_kernel<FvsrnDeviceVolume<16*3>>, stream, params.frame.size.x, params.frame.size.y, params); return; }
    if (WIDTH == 16*4) { FvsrnDeviceVolume<16*4> neuralvolume(network->get_network()); neuralvolume.launch(network_raymarching_kernel<FvsrnDeviceVolume<16*4>>, stream, params.frame.size.x, params.frame.size.y, params); return; }
    if (WIDTH == 16*5) { FvsrnDeviceVolume<16*5> neuralvolume(network->get_network()); neuralvolume.launch(network_raymarching_kernel<FvsrnDeviceVolume<16*5>>, stream, params.frame.size.x, params.frame.size.y, params); return; }
    if (WIDTH == 16*6) { FvsrnDeviceVolume<16*6> neuralvolume(network->get_network()); neuralvolume.launch(network_raymarching_kernel<FvsrnDeviceVolume<16*6>>, stream, params.frame.size.x, params.frame.size.y, params); return; }
    if (WIDTH == 16*7) { FvsrnDeviceVolume<16*7> neuralvolume(network->get_network()); neuralvolume.launch(network_raymarching_kernel<FvsrnDeviceVolume<16*7>>, stream, params.frame.size.x, params.frame.size.y, params); return; }
    if (WIDTH == 16*8) { FvsrnDeviceVolume<16*8> neuralvolume(network->get_network()); neuralvolume.launch(network_raymarching_kernel<FvsrnDeviceVolume<16*8>>, stream, params.frame.size.x, params.frame.size.y, params); return; }
    fprintf(stderr, "Unsupported MLP WIDTH for in-shader rendering: %d (%s: line %d)\n", WIDTH, __FILE__, __LINE__);
    return; 
  }
#endif

  if (N_FEATURES_PER_LEVEL == 1) return do_raymarching_network_template<1>(stream, params, *network);
  if (N_FEATURES_PER_LEVEL == 2) return do_raymarching_network_template<2>(stream, params, *network);
  if (N_FEATURES_PER_LEVEL == 4) return do_raymarching_network_template<4>(stream, params, *network);
  if (N_FEATURES_PER_LEVEL == 8) return do_raymarching_network_template<8>(stream, params, *network);

  fprintf(stderr, "Unsupported N_FEATURES_PER_LEVEL: %d (%s: line %d)\n", N_FEATURES_PER_LEVEL, __FILE__, __LINE__);
}

#endif // ENABLE_IN_SHADER

} // namespace vnr
