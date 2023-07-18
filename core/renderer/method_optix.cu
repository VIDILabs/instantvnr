//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //
//. ======================================================================== //
//. Copyright 2018-2019 Ingo Wald                                            //
//.                                                                          //
//. Licensed under the Apache License, Version 2.0 (the "License");          //
//. you may not use this file except in compliance with the License.         //
//. You may obtain a copy of the License at                                  //
//.                                                                          //
//.     http://www.apache.org/licenses/LICENSE-2.0                           //
//.                                                                          //
//. Unless required by applicable law or agreed to in writing, software      //
//. distributed under the License is distributed on an "AS IS" BASIS,        //
//. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
//. See the License for the specific language governing permissions and      //
//. limitations under the License.                                           //
//. ======================================================================== //

#include "method_optix.h"
#include "raytracing_shaders.h"

#include <optix.h>
#include <optix_device.h>

#include "dda.h"

#ifndef ADAPTIVE_SAMPLING
#error "ADAPTIVE_SAMPLING is not defined"
#endif

namespace vnr {

/*! launch parameters in constant memory, filled in by optix upon optixLaunch
    (this gets filled in from the buffer we pass to optixLaunch) */
extern "C" __constant__ DefaultUserData optixLaunchParams;

struct RadiancePayload
{
  float alpha = 0.f;
  vec3f color = 0.f;

  void* rng = nullptr;
  // float t_max = 0.f;
};

struct ShadowPayload
{
  float alpha = 0.f;

  void* rng = nullptr;
  // float t_max = 0.f;
};

//------------------------------------------------------------------------------
// helpers
// ------------------------------------------------------------------------------

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

inline __device__ float
transmittance(void* rng, const vec3f& org, const vec3f& dir)
{
  ShadowPayload shadow;
  shadow.rng = rng;
  // shadow.t_max = 0.f;

  uint32_t u0, u1;
  packPointer(&shadow, u0, u1);

  optixTrace(optixLaunchParams.traversable,
             /*org=*/org, /*dir=*/dir,
             /*tmin=*/0.f,
             /*tmax=*/float_large,
             /*time=*/0.0f,
             OptixVisibilityMask(255), /* not just volume */
             OPTIX_RAY_FLAG_DISABLE_ANYHIT,
             MethodOptiX::SHADOW_RAY_TYPE, // SBT offset
             MethodOptiX::RAY_TYPE_COUNT,  // SBT stride
             MethodOptiX::SHADOW_RAY_TYPE, // miss SBT index
             u0, u1);

  return 1.f - shadow.alpha;
}

//------------------------------------------------------------------------------
// helpers
// ------------------------------------------------------------------------------

inline __device__ void
shadeVolume_radiance(const DeviceVolume& self,
                     // world space ray
                     const vec3f _org,
                     const vec3f _dir,
                     const float tMin,
                     const float tMax,
                     // output
                     RadiancePayload& payload)
{
  const auto& params = optixLaunchParams;

  if (tMin >= tMax) {
    return;
  }

  const auto otw = getXfmOTW();
  const auto wto = getXfmWTO();
  const auto wtc = getXfmWorldToCamera(params);
  
  // to object space
  const vec3f org = xfmPoint(wto, _org);
  const vec3f dir = xfmVector(wto, _dir);

  // vec3f gradient = 0.f;
  vec3f highestContributionOrg = 0.f;
  vec3f highestContributionColor = 0.f;
  float highestContributionAlpha = 0.f;

  const auto& shadingScale = params.scivis_shading_scale;
  const auto& gradientStep = self.grad_step;

  auto rng = (RandomTEA*)payload.rng;
  const float jitter = rng->get_floats().x;

  // start marching 
  raymarching_iterator(self, org, dir, tMin, tMax, self.step, [&](const vec2f& t) 
  {
    // sample data value
    const auto p = org + lerp(jitter, t.x, t.y) * dir; // object space position
    const auto sampleValue = sampleVolume(self.volume, p);

    // classification
    vec3f sampleColor;
    float sampleAlpha;
    sampleTransferFunction(self.tfn, sampleValue, sampleColor, sampleAlpha);
    opacityCorrection(self, t.y - t.x, sampleAlpha);

    // sample gradient
    // (from OSPRay) assume that opacity directly correlates to volume scalar
    // field, i.e. that "outside" has lower values; because the gradient point
    // towards increasing values we need to flip it.
    // object space
    const vec3f No = -sampleGradient(self.volume, p, sampleValue, gradientStep);
    const vec3f Nw = xfmNormal(otw, No); // world  space
    // const vec3f Nc = xfmNormal(wtc, Nw); // camera space

    const float tr = 1.f - payload.alpha;

    // shade volume
    if (params.mode == MethodOptiX::GRADIENT_SHADING) {
      const vec3f shadingColor = shade_scivis_light(_dir, Nw, sampleColor, 
                                                    params.mat_gradient_shading, 
                                                    params.light_ambient, 
                                                    params.light_directional_rgb, 
                                                    params.light_directional_dir);
      sampleColor = lerp(shadingScale, sampleColor, shadingColor);
    }

    else if (params.mode == MethodOptiX::FULL_SHADOW) {
      const float shadow = transmittance(payload.rng, xfmPoint(otw, p), normalize(params.light_directional_dir));
      // const vec3f shadingColor = shade_scivis_light(_dir, Nw, sampleColor, 
      //                                               params.mat_full_shadow, 
      //                                               params.light_ambient, 
      //                                               params.light_directional_rgb, 
      //                                               params.light_directional_dir);
      sampleColor = lerp(shadingScale, sampleColor, sampleColor * shadow);
    }

    else if (params.mode == MethodOptiX::SINGLE_SHADE_HEURISTIC) {
      // remember point of highest density for deferred shading
      if (highestContributionAlpha < (1.f - payload.alpha) * sampleAlpha) {
        highestContributionOrg = p;
        highestContributionColor = sampleColor;
        highestContributionAlpha = (1.f - payload.alpha) * sampleAlpha;
      }
      // gradient += tr * Nw; // accumulate gradient for SSH
    }

    // blending
    payload.alpha += tr * sampleAlpha;
    payload.color += tr * sampleColor * sampleAlpha;

    return payload.alpha < nearly_one;
  });

  // At a single, point cast rays for contribution.
  if (highestContributionAlpha > 0.f) {
    const float shadow = transmittance(payload.rng, xfmPoint(otw, highestContributionOrg), normalize(params.light_directional_dir));
    // const vec3f shadingColor = shade_scivis_light(_dir, gradient, highestContributionColor, 
    //                                               params.mat_single_shade_heuristic, 
    //                                               params.light_ambient, 
    //                                               params.light_directional_rgb, 
    //                                               params.light_directional_dir);
    payload.color = lerp(shadingScale, payload.color, highestContributionColor * payload.alpha * shadow);
  }
}

inline __device__ void
shadeVolume_shadow(const DeviceVolume& self,
                   // world space ray
                   const vec3f _org,
                   const vec3f _dir,
                   const float tMin,
                   const float tMax,
                   // performance tuning
                   const float samplingScale,
                   // output
                   ShadowPayload& payload)
{
  if (tMin >= tMax) {
    return;
  }

  const Array3DScalar& volume = self.volume;
  const auto wto = getXfmWTO();

  // to object space
  const vec3f org = xfmPoint(wto, _org);
  const vec3f dir = xfmVector(wto, _dir);

  auto rng = (RandomTEA*)payload.rng;
  const float jitter = rng->get_floats().x;

  // start marching
  raymarching_iterator(self, org, dir, tMin, tMax, samplingScale * self.step, [&](const vec2f& t) 
  {
    // sample data value
    const auto p = org + lerp(jitter, t.x, t.y) * dir; // object space position
    const auto sampleValue = sampleVolume(volume, p);

    // classification
    vec3f sampleColor;
    float sampleAlpha;
    sampleTransferFunction(self.tfn, sampleValue, sampleColor, sampleAlpha);
    opacityCorrection(self, t.y - t.x, sampleAlpha);

    // blending 
    payload.alpha += (1.f - payload.alpha) * sampleAlpha;

    return payload.alpha < nearly_one;
  });
}

//------------------------------------------------------------------------------
// closesthit
// ------------------------------------------------------------------------------

extern "C" __global__ void
__closesthit__volume_radiance()
{
  const auto& self = getProgramData<DeviceVolume>();
  auto data = *getPRD<RadiancePayload>();

  const float t0 = optixGetRayTmax();
  const float t1 = __int_as_float(optixGetAttribute_1());

  const vec3f rayOrg = optixGetWorldRayOrigin();
  const vec3f rayDir = optixGetWorldRayDirection();

  // return pre-multiplied color
  shadeVolume_radiance(self, rayOrg, rayDir, t0, t1, data);

  // finalize
  // data.t_max = t1 + float_epsilon;
  *getPRD<RadiancePayload>() = data;
}

extern "C" __global__ void
__closesthit__volume_shadow()
{
  const auto& self = getProgramData<DeviceVolume>();
  auto data = *getPRD<ShadowPayload>();

  const float t0 = optixGetRayTmax();
  const float t1 = __int_as_float(optixGetAttribute_1());

  const vec3f rayOrg = optixGetWorldRayOrigin();
  const vec3f rayDir = optixGetWorldRayDirection();

  // return alpha
  shadeVolume_shadow(self, rayOrg, rayDir, t0, t1, optixLaunchParams.raymarching_shadow_sampling_scale, data);

  // finalize
  // data.t_max = t1 + float_epsilon;
  *getPRD<ShadowPayload>() = data;
}

//------------------------------------------------------------------------------
// anyhit
// ------------------------------------------------------------------------------

extern "C" __global__ void
__anyhit__volume_radiance()
{
}

extern "C" __global__ void
__anyhit__volume_shadow()
{
}

//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
//
// as with the anyhit/closest hit programs, in this example we only
// need to have _some_ dummy function to set up a valid SBT
// ------------------------------------------------------------------------------

extern "C" __global__ void
__miss__radiance()
{
  // RadiancePayload& payload = *getPRD<RadiancePayload>();
  // payload.t_max = optixGetRayTmax();
}

extern "C" __global__ void
__miss__shadow()
{
  // ShadowPayload& payload = *getPRD<ShadowPayload>();
  // payload.t_max = optixGetRayTmax();
}

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------

inline __device__ void
render_volume(vec3f org, vec3f dir, void* rng, float& _alpha, vec3f& _color)
{
  RadiancePayload payload;
  payload.rng = rng;

  uint32_t u0, u1;
  packPointer(&payload, u0, u1);

  /* trace non-volumes */
  // struct
  // {
  //   float alpha = 0.f;
  //   vec3f color = 0.f;
  // } background;

  // optixTrace(optixLaunchParams.traversable,
  //            /*org=*/org,
  //            /*dir=*/dir,
  //            /*tmin=*/0.f,
  //            /*tmax=*/float_large,
  //            /*time=*/0.0f,
  //            OptixVisibilityMask(~VISIBILITY_MASK_VOLUME), /* non-volume */
  //            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
  //            MethodOptiX::RADIANCE_RAY_TYPE, // SBT offset
  //            MethodOptiX::RAY_TYPE_COUNT,    // SBT stride
  //            MethodOptiX::RADIANCE_RAY_TYPE, // miss SBT index
  //            u0,
  //            u1);

  // background.alpha = payload.alpha;
  // background.color = payload.color;

  // payload.alpha = 0;
  // payload.color = 0;
  // payload.t_max = 0.f;

  /* trace volumes */
  optixTrace(optixLaunchParams.traversable,
             /*org=*/org,
             /*dir=*/dir,
             /*tmin=*/0.f,
             /*tmax=*/float_large,
             /*time=*/0.0f,
             OptixVisibilityMask(VISIBILITY_MASK_VOLUME), /* just volume */
             OPTIX_RAY_FLAG_DISABLE_ANYHIT,
             MethodOptiX::RADIANCE_RAY_TYPE, // SBT offset
             MethodOptiX::RAY_TYPE_COUNT,    // SBT stride
             MethodOptiX::RADIANCE_RAY_TYPE, // miss SBT index
             u0,
             u1);

  _alpha = payload.alpha;
  _color = payload.color;
}

extern "C" __global__ void
__raygen__default()
{
  // compute a test pattern based on pixel ID
  const int ix = optixGetLaunchIndex().x;
  const int iy = optixGetLaunchIndex().y;

  // pixel index
  const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;

  // random number generator
  RandomTEA rng_state(optixLaunchParams.frame_index, fbIndex);

  // normalized screen plane position, in [0,1]^2
  const auto& camera = optixLaunchParams.camera;
  const vec2f screen(vec2f((float)ix + .5f, (float)iy + .5f) / vec2f(optixLaunchParams.frame.size));

  // generate ray
  vec3f rayOrg = camera.position;
  vec3f rayDir = normalize(camera.direction +                      /* -z axis */
                           (screen.x - 0.5f) * camera.horizontal + /* x shift */
                           (screen.y - 0.5f) * camera.vertical);   /* y shift */

  // render
  float alpha = 0;
  vec3f color = 0;

  if (optixLaunchParams.mode == MethodOptiX::DEBUG) {
    RadiancePayload payload;
    payload.rng = (void*)&rng_state;

    uint32_t u0, u1;
    packPointer(&payload, u0, u1);

    optixTrace(optixLaunchParams.geometry_traversable,
               /**/ rayOrg,
               /**/ rayDir,
               /* tmin */ 0.f,
               /* tmax */ float_large,
               /* rayTime */ 0.0f,
               OptixVisibilityMask(255), // only have one volume
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               MethodOptiX::RADIANCE_RAY_TYPE, // SBT offset
               MethodOptiX::RAY_TYPE_COUNT,    // SBT stride
               MethodOptiX::RADIANCE_RAY_TYPE, // miss SBT index
               u0,
               u1);

    alpha = payload.alpha;
    color = payload.color;
  }

  else {
    render_volume(rayOrg, rayDir, (void*)&rng_state, alpha, color);
  }

  // and write to frame buffer ...
  writePixelColor(optixLaunchParams, vec4f(color, alpha), fbIndex);
}

} // namespace ovr
