#pragma once

#include "core/types.h"

#include <algorithm>

namespace ovr { 
namespace nnvolume {

using namespace vnr::math;

using vnr::LaunchParams;
using vnr::NeuralVolume;
using vnr::DeviceVolume;
using vnr::Array1D;
using vnr::Array3DScalar;
using vnr::DeviceTransferFunction;

inline __device__ bool
_intersectBox(float& _t0, float& _t1, const vec3f ray_org, const vec3f ray_dir, const vec3f lower, const vec3f upper)
{
  float t0 = _t0;
  float t1 = _t1;
#if 1
  const vec3i is_small = vec3i(fabs(ray_dir.x) <= float_small, fabs(ray_dir.y) <= float_small, fabs(ray_dir.z) <= float_small);
  const vec3f rcp_dir = vec3f(__frcp_rn(ray_dir.x), __frcp_rn(ray_dir.y), __frcp_rn(ray_dir.z));
  const vec3f t_lo = vec3f(is_small.x ? float_large : (lower.x - ray_org.x) * rcp_dir.x, //
                           is_small.y ? float_large : (lower.y - ray_org.y) * rcp_dir.y, //
                           is_small.z ? float_large : (lower.z - ray_org.z) * rcp_dir.z  //
  );
  const vec3f t_hi = vec3f(is_small.x ? -float_large : (upper.x - ray_org.x) * rcp_dir.x, //
                           is_small.y ? -float_large : (upper.y - ray_org.y) * rcp_dir.y, //
                           is_small.z ? -float_large : (upper.z - ray_org.z) * rcp_dir.z  //
  );
  t0 = max(t0, reduce_max(min(t_lo, t_hi)));
  t1 = min(t1, reduce_min(max(t_lo, t_hi)));
#else
  const vec3f t_lo = (lower - ray_org) / ray_dir;
  const vec3f t_hi = (upper - ray_org) / ray_dir;
  t0 = max(t0, reduce_max(min(t_lo, t_hi)));
  t1 = min(t1, reduce_min(max(t_lo, t_hi)));
#endif
  _t0 = t0;
  _t1 = t1;
  return t1 > t0;
}

inline __device__ bool
intersectVolume(float& _t0, float& _t1, const vec3f ray_org, const vec3f ray_dir, const DeviceVolume& volume)
{
  return _intersectBox(_t0, _t1, ray_org, ray_dir, volume.bbox.lower, volume.bbox.upper);
}

static __device__ affine3f
getXfmCameraToWorld(const LaunchParams& params)
{
  const auto& camera = params.camera;
  affine3f xfm;
  xfm.l.vx = normalize(camera.horizontal);
  xfm.l.vy = normalize(camera.vertical);
  xfm.l.vz = -normalize(camera.direction);
  xfm.p = camera.position;
  return xfm;
}

static __device__ affine3f
getXfmWorldToCamera(const LaunchParams& params)
{
  const auto& camera = params.camera;
  const auto x = normalize(camera.horizontal);
  const auto y = normalize(camera.vertical);
  const auto z = -normalize(camera.direction);
  affine3f xfm;
  xfm.l.vx = vec3f(x.x, y.x, z.x);
  xfm.l.vy = vec3f(x.y, y.y, z.y);
  xfm.l.vz = vec3f(x.z, y.z, z.z);
  xfm.p = -camera.position;
  return xfm;
}

template<typename T>
static __device__ T
array1dNodal(const Array1D& array, float v)
{
  if (array.length == 0) return T{0};

  assert(array.length > 0 && "invalid array size");
  v = clamp(v, 0.f, 1.f);
  const float t = fmaf(v, float(array.length-1), 0.5f) * __frcp_rn(array.length);
  return tex1D<T>(array.data, t);
}

static __device__ uint32_t
encodeVec4f(vec4f in)
{
  const uint32_t r(255.99f * clamp(in.x, 0.f, 1.f));
  const uint32_t g(255.99f * clamp(in.y, 0.f, 1.f));
  const uint32_t b(255.99f * clamp(in.z, 0.f, 1.f));
  const uint32_t a(255.99f * clamp(in.w, 0.f, 1.f));
  // convert to 32-bit rgba value (we explicitly set alpha to 0xff
  // to make stb_image_write happy) ...
  return (r << 0U) | (g << 8U) | (b << 16U) | (a << 24U);
}

static __device__ vec4f
decodeVec4f(uint32_t in)
{
  const float r = in >> 0U;
  const float g = in >> 8U;
  const float b = in >> 16U;
  const float a = in >> 24U;
  return vec4f(r, g, b, a) / 255.99f;
}

static __device__ float // it looks like we can only read textures as float
sampleVolume(const Array3DScalar& self, vec3f p)
{
  p = p * (1.f - self.rdims) + 0.5f * self.rdims;
  return tex3D<float>(self.data, p.x, p.y, p.z);
}

static __device__ vec3f
sampleGradient(const Array3DScalar& self,
               const vec3f c, // central position
               const float v, // central value
               vec3f stp)
{
  vec3f ext = c + stp;
  if (ext.x > 1.f-float_epsilon) stp.x *= -1.f;
  if (ext.y > 1.f-float_epsilon) stp.y *= -1.f;
  if (ext.z > 1.f-float_epsilon) stp.z *= -1.f;
  const vec3f gradient(sampleVolume(self, c + vec3f(stp.x, 0, 0)) - v,  //
                       sampleVolume(self, c + vec3f(0, stp.y, 0)) - v,  //
                       sampleVolume(self, c + vec3f(0, 0, stp.z)) - v); //
  return gradient / stp;
}

template<typename Sampler>
static __device__ vec3f
sampleGradient(const vec3f c, // central position
               const float v, // central value
               vec3f stp, 
               Sampler& sampler)
{
  vec3f ext = c + stp;
  if (ext.x > 1.f-float_epsilon) stp.x *= -1.f;
  if (ext.y > 1.f-float_epsilon) stp.y *= -1.f;
  if (ext.z > 1.f-float_epsilon) stp.z *= -1.f;
  const vec3f gradient(sampler(c + vec3f(stp.x, 0, 0)) - v,  //
                       sampler(c + vec3f(0, stp.y, 0)) - v,  //
                       sampler(c + vec3f(0, 0, stp.z)) - v); //
  return gradient / stp;
}

// it looks like we can only read textures as float

static __device__ void
sampleTransferFunction(const DeviceTransferFunction& tfn, float sampleValue, vec3f& _sampleColor, float& _sampleAlpha)
{
  const auto v = (clamp(sampleValue, tfn.range.lower, tfn.range.upper) - tfn.range.lower) * tfn.range_rcp_norm;
  vec4f rgba = array1dNodal<float4>(tfn.colors, v);
  rgba.w = array1dNodal<float>(tfn.alphas, v); // followed by the alpha correction
  _sampleColor = vec3f(rgba);
  _sampleAlpha = rgba.w;
}

static __device__ vec4f
sampleTransferFunction(const DeviceTransferFunction& tfn, float sampleValue)
{
  vec3f sampleColor;
  float sampleAlpha;
  sampleTransferFunction(tfn, sampleValue, sampleColor, sampleAlpha);
  return vec4f(sampleColor, sampleAlpha);
}

static __device__ void
opacityCorrection(const DeviceVolume& self, const float& distance, float& opacity)
{
  opacity = 1.f - __powf(1.f - opacity, 2.f * self.step_rcp * distance);
}

inline __device__ float
opacityUpperBound(const DeviceVolume& self, const vec3i& cell)
{
  const auto& dims = self.macrocell_dims;

  const uint32_t idx = cell.x + cell.y * uint32_t(dims.x) + cell.z * uint32_t(dims.x) * uint32_t(dims.y);
  assert(cell.x < dims.x);
  assert(cell.y < dims.y);
  assert(cell.z < dims.z);
  assert(cell.x >= 0);
  assert(cell.y >= 0);
  assert(cell.z >= 0);

  return self.macrocell_max_opacity[idx];
}

inline __device__ float
adaptiveSamplingRate(float base_sampling_step, float max_opacity)
{
  const float scale = 15 * base_sampling_step;
  const float r = fabsf(clamp(max_opacity, 0.1f, 1.f) - 1.f);
  return max(base_sampling_step + scale * std::pow(r, 2.f), base_sampling_step);
}

inline __device__ void
writePixelColor(const LaunchParams& params, const vec4f& color, const uint32_t pixel_index)
{
  vec4f rgba = color;
  if (params.frame_index == 1) {
    params.accumulation[pixel_index] = rgba;
  } else {
    rgba = params.accumulation[pixel_index] + rgba;
    params.accumulation[pixel_index] = rgba;
  }
  params.frame.rgba[pixel_index] = rgba / (float)params.frame_index;
}

// ------------------------------------------------------------------
// Helper Shading Functions
// Note: all inputs are world space vectors
// ------------------------------------------------------------------

inline __device__ vec3f
shade_simple_light(const vec3f& ray_dir, const vec3f& normal, const vec3f& albedo)
{
  if (dot(normal, normal) > 1.0e-6) {
    return albedo * (0.2f + .8f * fabsf(dot(-ray_dir, normalize(normal))));
  }
  return 0.f;
}

// inline __device__ vec3f
// shade_scivis_light(const vec3f& ray_dir, const vec3f& normal, const vec3f& albedo, const SciVisMaterial& mat,
//                    const vec3f& light_ambient, const vec3f& light_diffuse, const vec3f& light_dir)
// {
//   vec3f color = 0.f;
// 
//   if (dot(normal, normal) > 1.0e-6) {
//     const auto L = normalize(light_dir);
//     const auto N = normalize(normal);
//     const auto V = -ray_dir;
//     color += mat.ambient * albedo;
//     const float cosNL = std::max(dot(N, L), 0.f);
//     if (cosNL > 0.0f) {
//       color += mat.diffuse * cosNL * albedo * light_diffuse;
//       const vec3f H = normalize(L + V);
//       const float cosNH = std::max(dot(N, H), 0.f);
//       color += mat.specular * powf(cosNH, mat.shininess) * light_diffuse;
//     }
//   }
// 
//   const vec3f shading2 = shade_simple_light(ray_dir, normal, albedo);
// 
//   return lerp(0.5, shading2, color);
// }

// ------------------------------------------------------------------
// Additional Sampling Functions
// ------------------------------------------------------------------
__device__ constexpr float one_over_pi = 1.f / M_PI;
__device__ constexpr float four_pi = 4.f * M_PI;

inline __device__ vec3f
spherical_to_cartesian(const float phi, const float sinTheta, const float cosTheta)
{
  float sinPhi, cosPhi;
  sincosf(phi, &sinPhi, &cosPhi);
  return vec3f(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
}

inline __device__ vec3f
uniform_sample_sphere(const float radius, const vec2f s)
{
  const float phi = 2 * M_PI * s.x;
  const float cosTheta = radius * (1.f - 2.f * s.y);
  const float sinTheta = 2.f * radius * sqrt(s.y * (1.f - s.y));
  return spherical_to_cartesian(phi, sinTheta, cosTheta);
}

}
}
