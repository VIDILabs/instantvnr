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

#pragma once

#include "api.h"
// #include "object.h"
// #include "framebuffer.h"

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

struct TransferFunctionAPI
{
  DeviceTransferFunction tfn;
  cudaArray_t tfn_color_array_handler{};
  cudaArray_t tfn_alpha_array_handler{};
  ~TransferFunctionAPI();
  void update(const TransferFunction& tfn, const range1f original_data_range, cudaStream_t stream);
};

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

  void update(int rendering_mode, 
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

  void render(vec4f* fb, NeuralVolume* neuralnet, cudaTextureObject_t grid) {
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
};

#if 0
/*! a sample OptiX-7 renderer that demonstrates how to set up
    context, module, programs, pipeline, SBT, etc, and perform a
    valid launch that renders some pixel (using a simple test
    pattern, in this case */
struct MainRenderer
{
  // ------------------------------------------------------------------
  // publicly accessible interface
  // ------------------------------------------------------------------
public:
  ~MainRenderer();

  /*! constructor - performs all setup, including initializing
    optix, creates module, pipeline, programs, SBT, etc. */
  void init();

  /*! render one frame */
  void render();

  void set_scene(const cudaTextureObject_t& texture, 
                 ValueType type, vec3i dims, range1f range, 
                 affine3f transform, 
                 vec3i macrocell_dims, 
                 vec3f macrocell_spacing, 
                 vec2f* macrocell_d_value_range, 
                 float* macrocell_d_max_opacity,
                 NeuralVolume* neural_representation = nullptr);

  void set_scene_clipbox(const box3f& clip);

  void mapframe(vec4f** pixels)
  {
    CUDA_CHECK(cudaStreamSynchronize(framebuffer_stream));
    if (framebuffer_skip_download) {
      *pixels = framebuffer.device_pointer();
    }
    else {
      *pixels = framebuffer.host_pointer();
    }
    framebuffer.safe_swap();
  }

  /*! resize frame buffer to given resolution */
  void resize(const vec2i& new_size)
  {
    framebuffer_size = new_size;
    // resize our cuda frame buffer
    framebuffer.resize(new_size);
    reset_frame();
    // // update the launch parameters that we'll pass to the optix launch:
    // params.frame.size = framebuffer.size();
    // and re-set the camera, since aspect may have changed
    set_camera(camera_latest);
    // // resize auxiliary frame buffers
    // framebuffer_accumulation.resize(params.frame.size.long_product() * sizeof(vec4f), framebuffer_stream);
  }

  /*! set camera to render with */
  void set_camera(vec3f from, vec3f at, vec3f up) 
  { 
    set_camera(Camera{ from, at, up }); 
  }

  void set_camera(const Camera& camera)
  {
    camera_latest = camera;
    reset_frame();
  }

  void set_transfer_function(const std::vector<vec3f>& c, const std::vector<vec2f>& o, const range1f& r)
  {
    // volume.set_transfer_function(framebuffer_stream, c, o, r);

    std::vector<vec4f> colors_data;
    std::vector<float> alphas_data;

    colors_data.resize(c.size());
    for (int i = 0; i < colors_data.size(); ++i) {
      colors_data[i].x = c[i].x;
      colors_data[i].y = c[i].y;
      colors_data[i].z = c[i].z;
      colors_data[i].w = 1.f;
    }
    alphas_data.resize(o.size());
    for (int i = 0; i < alphas_data.size(); ++i) {
      alphas_data[i] = o[i].y;
    }

    if (!colors_data.empty())
      CreateArray1DFloat4(framebuffer_stream, colors_data, tfn_color_array_handler, tfn.colors);
    if (!alphas_data.empty())
      CreateArray1DScalar(framebuffer_stream, alphas_data, tfn_alpha_array_handler, tfn.alphas);

    // set_value_range(r.x, r.y);

    if (!r.is_empty()) {
      tfn.range.upper = min(original_data_range.upper, r.upper);
      tfn.range.lower = max(original_data_range.lower, r.lower);
    }
    tfn.range_rcp_norm = 1.f / tfn.range.span();
    
    reset_frame();
  }

  void set_volume_sampling_rate(float r)
  {
    // volume.set_sampling_rate(r);
    sampling_rate = r;
    reset_frame();
  }

  void set_volume_density_scale(float s)
  {
    // volume.set_density_scale(s);
    density_scale = s;
    reset_frame();
  }

  void set_rendering_mode(int b)
  {
    rendering_mode = b;
    reset_frame();

    // program_raymarching.clear(framebuffer_stream);
    // program_pathtracing.clear(framebuffer_stream);
  }

  void set_denoiser(bool enable)
  {
  }

  void set_output_as_cuda_framebuffer() { framebuffer_skip_download = true; }

  // const StructuredRegularVolume& get_volume() const { return volume; }
  // StructuredRegularVolume&       get_volume()       { return volume; }

  void reset_frame() { framebuffer_reset = true; }


  // ------------------------------------------------------------------
  // internal helper functions
  // ------------------------------------------------------------------
protected:
  /*! helper function that initializes optix and checks for errors */
  void initCuda();

  // /*! render volume */
  // void render_normal();
  // void render_neural();

public:
  // /*! @{ CUDA device context and stream that optix pipeline will run on, as well as device properties for this device */
  // cudaDeviceProp cuda_device_props{};
  // CUcontext cuda_context{};
  // cudaStream_t optix_default_stream{};
  // /*! @} */

  /*! @{ our launch parameters, on the host, and the buffer to store them on the device */
  // LaunchParams params;
  /*! @} */


  // MethodRayMarching program_raymarching;
  // MethodPathTracing program_pathtracing;

  // --------------------------------------------------------------- //
  // --------------------------------------------------------------- //
  int rendering_mode{ VNR_INVALID };

  NeuralVolume* neural_volume_representation{ nullptr };
  /*! we handle one volume and multiple geometries potentially */
  const cudaTextureObject_t* p_volume_data_texture{nullptr};
  cudaTextureObject_t volume_data_texture{ 0 };
  // StructuredRegularVolume volume;

  /*! the rendered image */
  FrameBuffer framebuffer;
  cudaStream_t framebuffer_stream{};
  bool framebuffer_reset{ true };
  bool framebuffer_skip_download{ false };
  // CUDABuffer framebuffer_accumulation;
  vec2i framebuffer_size;

  // volume states
  float sampling_rate{ 1.f };
  float density_scale{ 1.f };
  DeviceTransferFunction tfn;
  box3f bbox = box3f(vec3f(0), vec3f(1)); // object space box
  cudaArray_t tfn_color_array_handler{};
  cudaArray_t tfn_alpha_array_handler{};
  range1f original_data_range;


  /*! the camera we are to render with. */
  Camera camera_latest;

  // --------------------------------------------------------------- //
  // --------------------------------------------------------------- //
  RenderAPI ctx;
};

#endif

} // namespace vnr
