// credit to Ingo Wald

#pragma once

#include "core/types.h"

#include <gdt/math/vec.h>
#include <gdt/math/mat.h>

namespace dda {

using namespace gdt;

static inline __device__ vec3f
floor(vec3f v)
{
  return { floorf(v.x), floorf(v.y), floorf(v.z) };
}

struct DDAIter
{
  vec3f t_next;
  vec3i cell;
  float next_cell_begin = 0.f;

  void __device__ init(vec3f org, vec3f dir, float t_min, float t_max, vec3i grid_size) 
  {
    const float& ray_t0=t_min;
    const float& ray_t1=t_max;
    assert(ray_t0 < ray_t1);

    const vec3f org_in_volume = org + ray_t0 * dir;
    const vec3f f_cell = max(vec3f(0.f), min(vec3f(grid_size) - 1.f, floor(org_in_volume)));
    const vec3f f_cell_end = {
      dir.x > 0.f ? f_cell.x + 1.f : f_cell.x,
      dir.y > 0.f ? f_cell.y + 1.f : f_cell.y,
      dir.z > 0.f ? f_cell.z + 1.f : f_cell.z,
    };
    const vec3f t_step = abs(rcp(dir));

    t_next = { ((dir.x == 0.f) ? float_large : (abs(f_cell_end.x - org_in_volume.x) * t_step.x)),
               ((dir.y == 0.f) ? float_large : (abs(f_cell_end.y - org_in_volume.y) * t_step.y)),
               ((dir.z == 0.f) ? float_large : (abs(f_cell_end.z - org_in_volume.z) * t_step.z)) };
    cell = vec3i(f_cell);
    next_cell_begin = 0.f;
  }

  template<typename Lambda>
  bool __device__ next(vec3f org, vec3f dir, float t_min, float t_max, vec3i grid_size, bool dbg, const Lambda& lambda)
  {
    // if (dbg)
    //   printf("cell %d %d %d\n",
    //         cell.x,
    //         cell.y,
    //         cell.z);
    // if (dbg)
    //   printf("t_next %f %f %f\n",
    //         t_next.x,
    //         t_next.y,
    //         t_next.z);
    // if (dbg)
    //   printf("next_cell_begin %f\n", next_cell_begin);

    const vec3i stop = { dir.x > 0.f ? (int)grid_size.x : -1, dir.y > 0.f ? (int)grid_size.y : -1, dir.z > 0.f ? (int)grid_size.z : -1 };
    if (cell.x == stop.x) return false;
    if (cell.y == stop.y) return false;
    if (cell.z == stop.z) return false;
    // assert(cell.x != stop.x);
    // assert(cell.y != stop.y);
    // assert(cell.z != stop.z);

    const float& ray_t0=t_min;
    const float& ray_t1=t_max;
    assert(ray_t0 < ray_t1);
    // if (dbg) printf("t range for volume %f %f\n",ray_t0,ray_t1); 

    const vec3f t_step = abs(rcp(dir));
    const vec3i cell_delta = { (dir.x > 0.f ? +1 : -1), (dir.y > 0.f ? +1 : -1), (dir.z > 0.f ? +1 : -1) };
    // if (dbg) printf("stop %d %d %d\n", stop.x, stop.y, stop.z);
    // if (dbg) printf("t_step %f %f %f\n", t_step.x, t_step.y, t_step.z);

    const float t_closest = reduce_min(t_next);
    const float cell_t0 = max(ray_t0 + next_cell_begin, t_min);
    const float cell_t1 = min(ray_t0 + t_closest, t_max);
    // if (dbg)
    //   printf("cell %i %i %i dists %f %f %f closest %f t %f %f\n",
    //          cell.x,cell.y,cell.z,
    //          t_next.x,t_next.y,t_next.z,
    //          t_closest,cell_t0,cell_t1);
    if (cell_t0 >= cell_t1)
      return false;

    const bool goto_next_cell = lambda(cell, cell_t0, cell_t1);

    // we should update `next_cell_begin` in the lambda function. We do not want to 
    // update it here if the loop is terminated by the lambda function.
    if (goto_next_cell || max(t_min + next_cell_begin, t_min) >= cell_t1) {
      if (t_next.x == t_closest) {
        t_next.x += t_step.x;
        cell.x += cell_delta.x;
        if (cell.x == stop.x) return false;
      }
      if (t_next.y == t_closest) {
        t_next.y += t_step.y;
        cell.y += cell_delta.y;
        if (cell.y == stop.y) return false;
      }
      if (t_next.z == t_closest) {
        t_next.z += t_step.z;
        cell.z += cell_delta.z;
        if (cell.z == stop.z) return false;
      }
      next_cell_begin = t_closest;
    }

    // if (dbg)
    //   printf("post cell %d %d %d  stop %d %d %d\n",
    //          cell.x, cell.y, cell.z,
    //          stop.x, stop.y, stop.z);

    return goto_next_cell;
  }

  bool __device__ resumable(vec3f dir, float t_min, float t_max, vec3i grid_size) const
  {
    const vec3i stop = { dir.x > 0.f ? (int)grid_size.x : -1, dir.y > 0.f ? (int)grid_size.y : -1, dir.z > 0.f ? (int)grid_size.z : -1 };
    if (cell.x == stop.x) return false;
    if (cell.y == stop.y) return false;
    if (cell.z == stop.z) return false;
    const float t_closest = reduce_min(t_next);
    const float cell_t0 = max(t_min + next_cell_begin, t_min);
    const float cell_t1 = min(t_min + t_closest, t_max);
    if (cell_t0 >= cell_t1) {
      return false;
    }
    return true;
  }
};

template<typename Lambda>
inline __device__ void
dda3(vec3f org, vec3f dir, float t_min, float t_max, vec3i grid_size, bool dbg, const Lambda& lambda)
{
  // const box3f bounds = { vec3f(0.f), vec3f(grid_size) };
  // const vec3f floor_org = floor(org);
  // const vec3f floor_org_plus_one = floor_org + vec3f(1.f);
  // const vec3f rcp_dir = rcp(dir);
  // const vec3f abs_rcp_dir = abs(rcp(dir));
  // const vec3f f_size = vec3f(grid_size);

  // vec3f t_lo = (vec3f(0.f) - org) * rcp(dir);
  // vec3f t_hi = (f_size - org) * rcp(dir);
  // vec3f t_nr = min(t_lo, t_hi);
  // vec3f t_fr = max(t_lo, t_hi);
  // if (dir.x == 0.f) {
  //   if (org.x <= 0.f || org.x >= f_size.x) {
  //     return; // ray passes by the volume ...
  //   }
  //   t_nr.x = -float_large;
  //   t_fr.x = +float_large;
  // }
  // if (dir.y == 0.f) {
  //   if (org.y <= 0.f || org.y >= f_size.y) {
  //     return; // ray passes by the volume ...
  //   }
  //   t_nr.y = -float_large;
  //   t_fr.y = +float_large;
  // }
  // if (dir.z == 0.f) {
  //   if (org.z <= 0.f || org.z >= f_size.z) {
  //     return; // ray passes by the volume ...
  //   }
  //   t_nr.z = -float_large;
  //   t_fr.z = +float_large;
  // }

  // float ray_t0 = max(t_min, reduce_max(t_nr));
  // float ray_t1 = min(t_max, reduce_min(t_fr));
  const float& ray_t0=t_min;
  const float& ray_t1=t_max;

  // if (isnan(ray_t0) || isnan(ray_t1))
  //   printf("NAN in DDA!\n");
  // if (dbg) printf("t range for volume %f %f\n",ray_t0,ray_t1);
  if (ray_t0 >= ray_t1) return; // no overlap with volume

  // compute first cell that ray is in:
  const vec3f org_in_volume = org + ray_t0 * dir;
  // if (dbg) printf("org in vol %f %f %f size %i %i %i\n",
  //                 org_in_volume.x,
  //                 org_in_volume.y,
  //                 org_in_volume.z,
  //                 grid_size.x,
  //                 grid_size.y,
  //                 grid_size.z);
  // if (dbg) printf("org %f %f %f\n", org.x, org.y, org.z);
  // if (dbg) printf("dir %f %f %f\n", dir.x, dir.y, dir.z);
  vec3f f_cell = max(vec3f(0.f), min(vec3f(grid_size) - 1.f, floor(org_in_volume)));
  vec3f f_cell_end = {
    dir.x > 0.f ? f_cell.x + 1.f : f_cell.x,
    dir.y > 0.f ? f_cell.y + 1.f : f_cell.y,
    dir.z > 0.f ? f_cell.z + 1.f : f_cell.z,
  };
  // vec3f f_cell_end = {
  //   dir.x > 0.f ? f_cell.x + 1.f : (org_in_volume.x > floorf(org_in_volume.x) ?  f_cell.x : f_cell.x - 1.f),
  //   dir.y > 0.f ? f_cell.y + 1.f : (org_in_volume.y > floorf(org_in_volume.y) ?  f_cell.y : f_cell.y - 1.f),
  //   dir.z > 0.f ? f_cell.z + 1.f : (org_in_volume.z > floorf(org_in_volume.z) ?  f_cell.z : f_cell.z - 1.f),
  // };
  // if (dbg)
  //   printf("f_cell %f %f %f\n", f_cell.x, f_cell.y, f_cell.z);
  // if (dbg)
  //   printf("f_cell_end %f %f %f\n",
  //          f_cell_end.x,
  //          f_cell_end.y,
  //          f_cell_end.z);

  const vec3f t_step = abs(rcp(dir));
  // if (dbg)
  //   printf("t_step %f %f %f\n",
  //          t_step.x,
  //          t_step.y,
  //          t_step.z);
  vec3f t_next = { ((dir.x == 0.f) ? float_large : (abs(f_cell_end.x - org_in_volume.x) * t_step.x)),
                   ((dir.y == 0.f) ? float_large : (abs(f_cell_end.y - org_in_volume.y) * t_step.y)),
                   ((dir.z == 0.f) ? float_large : (abs(f_cell_end.z - org_in_volume.z) * t_step.z)) };
  // if (dbg)
  //   printf("t_next %f %f %f\n",
  //          t_next.x,
  //          t_next.y,
  //          t_next.z);
  const vec3i stop = { dir.x > 0.f ? (int)grid_size.x : -1, dir.y > 0.f ? (int)grid_size.y : -1, dir.z > 0.f ? (int)grid_size.z : -1 };
  // if (dbg)
  //   printf("stop %i %i %i\n",
  //          stop.x,
  //          stop.y,
  //          stop.z);
  const vec3i cell_delta = { (dir.x > 0.f ? +1 : -1), (dir.y > 0.f ? +1 : -1), (dir.z > 0.f ? +1 : -1) };
  // if (dbg)
  //   printf("cell_delta %i %i %i\n",
  //          cell_delta.x,
  //          cell_delta.y,
  //          cell_delta.z);
  vec3i cell = vec3i(f_cell);
  float next_cell_begin = 0.f;

  while (true) {
    const float t_closest = reduce_min(t_next);

    const float cell_t0 = max(ray_t0 + next_cell_begin, t_min);
    const float cell_t1 = min(ray_t0 + t_closest, t_max);
    // const double cell_t0 = max((double)ray_t0 + next_cell_begin, (double)t_min);
    // const double cell_t1 = min((double)ray_t0 + t_closest, (double)t_max);
    // if (dbg)
    //   printf("cell %i %i %i dists %f %f %f closest %f t %f %f\n",
    //          cell.x,cell.y,cell.z,
    //          t_next.x,t_next.y,t_next.z,
    //          t_closest,cell_t0,cell_t1);
    
    if (cell_t0 >= cell_t1)
      return;

    // vec3f org_0 = org + (float)cell_t0 * dir;
    // vec3f org_1 = org + (float)cell_t1 * dir;
    // if (dbg) printf("org_0 (%f %f %f) org_1 (%f %f %f)\n", org_0.x, org_0.y, org_0.z, org_1.x, org_1.y, org_1.z);

    bool want_to_go_on = lambda(cell, cell_t0, cell_t1);

    if (!want_to_go_on) return;

    if (t_next.x == t_closest) {
      t_next.x += t_step.x;
      cell.x += cell_delta.x;
      if (cell.x == stop.x) return;
    }
    if (t_next.y == t_closest) {
      t_next.y += t_step.y;
      cell.y += cell_delta.y;
      if (cell.y == stop.y) return;
    }
    if (t_next.z == t_closest) {
      t_next.z += t_step.z;
      cell.z += cell_delta.z;
      if (cell.z == stop.z) return;
    }
    next_cell_begin = t_closest;
  }
}

template<typename Lambda>
inline __device__ void
network_dda3(vec3f org, vec3f dir, float t_min, float t_max, vec3i grid_size, bool dbg, const Lambda& lambda)
{
  // const vec3f floor_org = floor(org);
  // const vec3f floor_org_plus_one = floor_org + vec3f(1.f);
  // const vec3f rcp_dir = rcp(dir);
  // const vec3f abs_rcp_dir = abs(rcp(dir));
  // const vec3f f_size = vec3f(grid_size);

  // compute first cell that ray is in:
  const vec3f org_in_volume = org + t_min * dir;
  vec3f f_cell = max(vec3f(0.f), min(vec3f(grid_size) - 1.f, floor(org_in_volume)));
  vec3f f_cell_end = {
    dir.x > 0.f ? f_cell.x + 1.f : f_cell.x,
    dir.y > 0.f ? f_cell.y + 1.f : f_cell.y,
    dir.z > 0.f ? f_cell.z + 1.f : f_cell.z,
  };
  // vec3f f_cell_end = {
  //   dir.x > 0.f ? f_cell.x + 1.f : (org_in_volume.x > floorf(org_in_volume.x) ?  f_cell.x : f_cell.x - 1.f),
  //   dir.y > 0.f ? f_cell.y + 1.f : (org_in_volume.y > floorf(org_in_volume.y) ?  f_cell.y : f_cell.y - 1.f),
  //   dir.z > 0.f ? f_cell.z + 1.f : (org_in_volume.z > floorf(org_in_volume.z) ?  f_cell.z : f_cell.z - 1.f),
  // };
  const vec3f t_step = abs(rcp(dir));
  vec3f t_next = { ((dir.x == 0.f) ? float_large : (abs(f_cell_end.x - org_in_volume.x) * t_step.x)),
                   ((dir.y == 0.f) ? float_large : (abs(f_cell_end.y - org_in_volume.y) * t_step.y)),
                   ((dir.z == 0.f) ? float_large : (abs(f_cell_end.z - org_in_volume.z) * t_step.z)) };
  const vec3i stop = { dir.x > 0.f ? (int)grid_size.x : -1, dir.y > 0.f ? (int)grid_size.y : -1, dir.z > 0.f ? (int)grid_size.z : -1 };
  const vec3i cell_delta = { (dir.x > 0.f ? +1 : -1), (dir.y > 0.f ? +1 : -1), (dir.z > 0.f ? +1 : -1) };
  vec3i cell = vec3i(f_cell);
  float next_cell_begin = 0.f;

  bool alive = (t_min < t_max);
  while (vnr::block_any(alive)) {
    float t_closest = reduce_min(t_next);

    const float cell_t0 = max(t_min + next_cell_begin, t_min);
    const float cell_t1 = min(t_min + t_closest, t_max);
    // const double cell_t0 = max((double)t_min + next_cell_begin, (double)t_min);
    // const double cell_t1 = min((double)t_min + t_closest, (double)t_max);
    if (cell_t0 >= cell_t1)
      alive = false;

    bool want_to_go_on = lambda(cell, cell_t0, cell_t1, alive);

    if (alive) {
      if (!want_to_go_on)
        alive = false;
      if (t_next.x == t_closest) {
        t_next.x += t_step.x;
        cell.x += cell_delta.x;
        if (cell.x == stop.x)
          alive = false;
      }
      if (t_next.y == t_closest) {
        t_next.y += t_step.y;
        cell.y += cell_delta.y;
        if (cell.y == stop.y)
          alive = false;
      }
      if (t_next.z == t_closest) {
        t_next.z += t_step.z;
        cell.z += cell_delta.z;
        if (cell.z == stop.z)
          alive = false;
      }
      next_cell_begin = t_closest;
    }
  }
}

} // namespace dda
