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

#include "renderer.h"

#include <iostream>

#ifdef ENABLE_LOGGING
#define log() std::cout
#else
static std::ostream null_output_stream(0);
#define log() null_output_stream
#endif

namespace vnr {

// TransferFunctionAPI::~TransferFunctionAPI() {
//   if (tfn_color_array_handler) {
//     CUDA_CHECK_NOEXCEPT(cudaTrackedFreeArray(tfn_color_array_handler));
//     tfn_color_array_handler = NULL;
//   }
//   if (tfn.colors.data) {
//     CUDA_CHECK_NOEXCEPT(cudaDestroyTextureObject(tfn.colors.data));
//     tfn.colors.data = { 0 };
//   }
//   if (tfn.colors.rawptr) {
//     CUDA_CHECK_NOEXCEPT(cudaTrackedFree(tfn.colors.rawptr, tfn.colors.length * sizeof(float4)));
//     tfn.colors.rawptr = nullptr;
//   }
//   tfn.colors.length = 0;

//   if (tfn_alpha_array_handler) {
//     CUDA_CHECK_NOEXCEPT(cudaTrackedFreeArray(tfn_alpha_array_handler));
//     tfn_color_array_handler = NULL;
//   }
//   if (tfn.alphas.data) {
//     CUDA_CHECK_NOEXCEPT(cudaDestroyTextureObject(tfn.alphas.data));
//     tfn.alphas.data = { 0 };
//   }
//   if (tfn.alphas.rawptr) {
//     CUDA_CHECK_NOEXCEPT(cudaTrackedFree(tfn.alphas.rawptr, tfn.alphas.length * sizeof(float)));
//     tfn.alphas.rawptr = nullptr;
//   }
//   tfn.alphas.length = 0;
// }

// void TransferFunctionAPI::update(const TransferFunction& input, const range1f original_data_range, cudaStream_t framebuffer_stream) {
//   const std::vector<vec3f>& c = input.color;
//   const std::vector<vec2f>& o = input.alpha;
//   const range1f& r = input.range;

//   std::vector<vec4f> colors_data;
//   std::vector<float> alphas_data;
//   colors_data.resize(c.size());
//   for (int i = 0; i < colors_data.size(); ++i) {
//     colors_data[i].x = c[i].x;
//     colors_data[i].y = c[i].y;
//     colors_data[i].z = c[i].z;
//     colors_data[i].w = 1.f;
//   }
//   alphas_data.resize(o.size());
//   for (int i = 0; i < alphas_data.size(); ++i) {
//     alphas_data[i] = o[i].y;
//   }
//   if (!colors_data.empty()) {  
//     CreateArray1DFloat4(framebuffer_stream, colors_data, tfn_color_array_handler, tfn.colors);
//   }
//   if (!alphas_data.empty()) {
//     CreateArray1DScalar(framebuffer_stream, alphas_data, tfn_alpha_array_handler, tfn.alphas);
//   }
//   if (!r.is_empty()) {
//     tfn.range.upper = min(original_data_range.upper, r.upper);
//     tfn.range.lower = max(original_data_range.lower, r.lower);
//   }
//   tfn.range_rcp_norm = 1.f / tfn.range.span();
// }

void RenderAPI::init(affine3f transform, 
  ValueType type, vec3i dims, range1f range, 
  vec3i macrocell_dims, 
  vec3f macrocell_spacings, 
  vec2f* macrocell_d_value_range, 
  float* macrocell_d_max_opacity
) {
  device_buffer.alloc(sizeof(self), stream);
  params.transform = transform;
  self.volume.dims = dims;
  self.volume.type = type;
  self.macrocell_value_range = macrocell_d_value_range;
  self.macrocell_max_opacity = macrocell_d_max_opacity;
  self.macrocell_dims = macrocell_dims;
  self.macrocell_spacings = macrocell_spacings;
  self.macrocell_spacings_rcp = 1.f / macrocell_spacings;
}

void RenderAPI::update(int rendering_mode, 
  const DeviceTransferFunction& tfn,
  float sampling_rate,
  float density_scale,
  vec3f clip_lower, 
  vec3f clip_upper,
  const Camera& camera,
  const vec2i& framesize
) {
  if (this->rendering_mode != rendering_mode) {
    this->rendering_mode = rendering_mode;
    program_raymarching.clear(stream);
    program_pathtracing.clear(stream);
  }

  self.step = 1.f / sampling_rate;
  self.step_rcp = sampling_rate;
  self.grad_step = vec3f(1.f / vec3f(self.volume.dims));
  self.density_scale = density_scale;
  self.tfn = tfn;
  self.tfn.range_rcp_norm = 1.f / self.tfn.range.span();
  self.bbox.lower = clip_lower;
  self.bbox.upper = clip_upper;

  // resize our cuda frame buffer
  framebuffer_accumulation.resize(framesize.long_product() * sizeof(vec4f), stream);
  params.frame.size = framesize;
  params.accumulation = (vec4f*)framebuffer_accumulation.d_pointer();

  /* camera ... */
  /* the factor '2.f' here might be unnecessary, but I want to match ospray's implementation */
  const float fovy = camera.fovy;
  const float t = 2.f /* (note above) */ * tan(fovy * 0.5f * (float)M_PI / 180.f);
  const float aspect = params.frame.size.x / float(params.frame.size.y);
  params.last_camera = params.camera;
  params.camera.position = camera.from;
  params.camera.direction = normalize(camera.at - camera.from);
  params.camera.horizontal = t * aspect * normalize(cross(params.camera.direction, camera.up));
  params.camera.vertical = cross(params.camera.horizontal, params.camera.direction) / aspect;
  /* correct light direction */
  if (dot(params.camera.direction, params.light_directional_dir) > 0) {
    params.light_directional_dir *= -1;
  }

  // flag to reset frame data
  params.frame_index = 0;
}

void RenderAPI::render(vec4f* fb, NeuralVolume* neuralnet, cudaTextureObject_t grid) {
  // set volume data
  self.volume.data = grid;
  if (!neuralnet && self.volume.data == 0) {
    std::cerr << "WARNING: no volume data to render" << std::endl; return;
  }
  // upload to GPU4
  device_buffer.upload_async(&self, 1, stream);
  DeviceVolume* dptr = (DeviceVolume*)device_buffer.d_pointer();
  /* rendering */
  params.frame_index++;
  params.frame.rgba = fb;
  switch (rendering_mode) {
  // path tracing
  case VNR_PATHTRACING_DECODING:         program_pathtracing.render(stream, params, dptr);                   break;
  case VNR_PATHTRACING_SAMPLE_STREAMING: program_pathtracing.render(stream, params, dptr, neuralnet, true);  break;
  case VNR_PATHTRACING_IN_SHADER:        program_pathtracing.render(stream, params, dptr, neuralnet, false); break;
  // ray marching
  case VNR_RAYMARCHING_NO_SHADING_DECODING:         program_raymarching.render(stream, params, MethodRayMarching::NO_SHADING, dptr);                   break;
  case VNR_RAYMARCHING_NO_SHADING_SAMPLE_STREAMING: program_raymarching.render(stream, params, MethodRayMarching::NO_SHADING, dptr, neuralnet, true);  break;
  case VNR_RAYMARCHING_NO_SHADING_IN_SHADER:        program_raymarching.render(stream, params, MethodRayMarching::NO_SHADING, dptr, neuralnet, false); break;
  // ray marching
  case VNR_RAYMARCHING_GRADIENT_SHADING_DECODING:         program_raymarching.render(stream, params, MethodRayMarching::GRADIENT_SHADING, dptr);                   break;
  case VNR_RAYMARCHING_GRADIENT_SHADING_SAMPLE_STREAMING: program_raymarching.render(stream, params, MethodRayMarching::GRADIENT_SHADING, dptr, neuralnet, true);  break;
  case VNR_RAYMARCHING_GRADIENT_SHADING_IN_SHADER:        program_raymarching.render(stream, params, MethodRayMarching::GRADIENT_SHADING, dptr, neuralnet, false); break;
  // ray marching
  case VNR_RAYMARCHING_SINGLE_SHADE_HEURISTIC_DECODING:         program_raymarching.render(stream, params, MethodRayMarching::SINGLE_SHADE_HEURISTIC, dptr);                   break;
  case VNR_RAYMARCHING_SINGLE_SHADE_HEURISTIC_SAMPLE_STREAMING: program_raymarching.render(stream, params, MethodRayMarching::SINGLE_SHADE_HEURISTIC, dptr, neuralnet, true);  break;
  case VNR_RAYMARCHING_SINGLE_SHADE_HEURISTIC_IN_SHADER:        program_raymarching.render(stream, params, MethodRayMarching::SINGLE_SHADE_HEURISTIC, dptr, neuralnet, false); break;
  default: break;
  }
}

} // namespace vnr
