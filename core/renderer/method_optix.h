//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //

#pragma once

#include "optix_program.h"

#include <cuda/cuda_buffer.h>

namespace vnr {

class MethodOptiX : public OptixProgram
{
public:
  enum { RADIANCE_RAY_TYPE = 0, SHADOW_RAY_TYPE, RAY_TYPE_COUNT };

  enum ShadingMode { NO_SHADING = 0, GRADIENT_SHADING, FULL_SHADOW, SINGLE_SHADE_HEURISTIC, DEBUG };

  MethodOptiX();
  ~MethodOptiX() { params_buffer.free(0); }
  void render(cudaStream_t stream, const LaunchParams& params, ShadingMode mode);

private:
  CUDABuffer params_buffer;
};

struct DefaultUserData  : LaunchParams
{
  DefaultUserData(const LaunchParams& p) : LaunchParams(p) {}

  MethodOptiX::ShadingMode mode;
  OptixTraversableHandle traversable{};
  OptixTraversableHandle geometry_traversable{};
};

}; // namespace ovr
