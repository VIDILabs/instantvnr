//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //
#pragma once

#include "raytracing.h"
#include <optix_device.h>

namespace vnr {

static __forceinline__ __device__ void*
unpackPointer(uint32_t i0, uint32_t i1)
{
  const auto uptr = static_cast<uint64_t>(i0) << 32U | i1;
  void* ptr = reinterpret_cast<void*>(uptr);
  return ptr;
}

static __forceinline__ __device__ void
packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
{
  const auto uptr = reinterpret_cast<uint64_t>(ptr);
  i0 = uptr >> 32U;
  i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T*
getPRD()
{
  const uint32_t u0 = optixGetPayload_0();
  const uint32_t u1 = optixGetPayload_1();
  return reinterpret_cast<T*>(unpackPointer(u0, u1));
}

template<typename T>
__device__ inline const T&
getProgramData()
{
  return **((const T**)optixGetSbtDataPointer());
}

inline __device__ affine3f
getXfmWTO()
{
  float mat[12]; /* 3x4 row major */
  optixGetWorldToObjectTransformMatrix(mat);
  affine3f xfm;
  xfm.l.vx = vec3f(mat[0], mat[4], mat[8]);
  xfm.l.vy = vec3f(mat[1], mat[5], mat[9]);
  xfm.l.vz = vec3f(mat[2], mat[6], mat[10]);
  xfm.p = vec3f(mat[3], mat[7], mat[11]);
  return xfm;
}

inline __device__ affine3f
getXfmOTW()
{
  float mat[12]; /* 3x4 row major */
  optixGetObjectToWorldTransformMatrix(mat);
  affine3f xfm;
  xfm.l.vx = vec3f(mat[0], mat[4], mat[8]);
  xfm.l.vy = vec3f(mat[1], mat[5], mat[9]);
  xfm.l.vz = vec3f(mat[2], mat[6], mat[10]);
  xfm.p = vec3f(mat[3], mat[7], mat[11]);
  return xfm;
}

inline vec2f __device__
projectToScreen(const vec3f p, const DeviceCamera& camera)
{
  vec3f wsvec = p - camera.position;
  vec2f screen;
  const float r = length(camera.horizontal);
  const float t = length(camera.vertical);
  screen.x = dot(wsvec, normalize(camera.horizontal)) / r;
  screen.y = dot(wsvec, normalize(camera.vertical)) / t;
  return screen + 0.5f;
}

//------------------------------------------------------------------------------
// intersection program that computes customized intersections for an AABB
// ------------------------------------------------------------------------------

extern "C" __global__ void
__intersection__volume()
{
  const vec3f org = optixGetObjectRayOrigin();
  const vec3f dir = optixGetObjectRayDirection();

  float t0 = optixGetRayTmin();
  float t1 = optixGetRayTmax();

  const auto& self = getProgramData<DeviceVolume>();

  if (intersectVolume(t0, t1, org, dir, self)) {
    optixReportIntersection(t0, 0, /* user defined attributes, for now set to 0 */
                            __float_as_int(t0), __float_as_int(t1));
  }
}

}
