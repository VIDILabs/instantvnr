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

// ----------------------------------------------------------------------------
//  renderer.h
//
//  `RenderAPI` - host-side front-end to the CUDA rendering kernels. It
//  bundles:
//
//    * `params`                   `LaunchParams` blob passed to every
//                                 kernel (camera, lights, frame index, ...).
//    * `self`                     `DeviceVolume` describing the volume
//                                 being rendered (grid dims, macrocells,
//                                 bbox, stepping, ...).
//    * `program_raymarching`      ray marching method dispatcher
//                                 (`renderer/method_raymarching.*`).
//    * `program_pathtracing`      volumetric path tracing dispatcher
//                                 (`renderer/method_pathtracing.*`).
//    * `framebuffer_accumulation` per-pixel accumulator for progressive
//                                 refinement.
//
//  Usage:
//      api.init(...);                           // allocate state once
//      api.update(mode, tfn, rate, scale, ...); // update per frame
//      api.render(fb, neuralnet, grid_tex);     // launch kernels
//
//  The active `mode` picks one of the `vnrRenderMode` values defined in
//  `api.h`; see the matrix at the top of `renderer/method_raymarching.cu`
//  and `renderer/method_pathtracing.cu` for how each mode is implemented.
// ----------------------------------------------------------------------------

#pragma once

#include "api.h"
#include "core/renderer/method_raymarching.h"
#include "core/renderer/method_pathtracing.h"

#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <map>
#include <vector>

namespace vnr {

// ------------------------------------------------------------------
// I/O helper functions
// ------------------------------------------------------------------

// struct TransferFunctionAPI
// {
//   DeviceTransferFunction tfn;
//   cudaArray_t tfn_color_array_handler{};
//   cudaArray_t tfn_alpha_array_handler{};
//   ~TransferFunctionAPI();
//   void update(const TransferFunction& tfn, const range1f original_data_range, cudaStream_t stream);
// };

typedef TransferFunctionObject TransferFunctionAPI;

struct RenderAPI {
  LaunchParams params;
  DeviceVolume self;
  CUDABuffer device_buffer;
  MethodRayMarching program_raymarching;
  MethodPathTracing program_pathtracing;

  // --------------------------------------------------------------- //
  // --------------------------------------------------------------- //
  int rendering_mode{ VNR_INVALID };
  CUDABuffer framebuffer_accumulation;
  cudaStream_t stream{ nullptr };

  void init(
    affine3f transform, 
    ValueType type, vec3i dims, range1f range, 
    vec3i macrocell_dims, 
    vec3f macrocell_spacings, 
    vec2f* macrocell_d_value_range, 
    float* macrocell_d_max_opacity
  );

  void update(int rendering_mode, 
    const DeviceTransferFunction& tfn,
    float sampling_rate,
    float density_scale,
    vec3f clip_lower, 
    vec3f clip_upper,
    const Camera& camera,
    const vec2i& framesize
  );

  void render(vec4f* fb, NeuralVolume* neuralnet, cudaTextureObject_t grid);
};

} // namespace vnr
