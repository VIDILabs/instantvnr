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
#include "object.h"
#include "framebuffer.h"

#if defined(ENABLE_OPTIX)
#include "core/renderer/method_optix.h"
#endif
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
  ~MainRenderer()
  {
    framebuffer_accumulation.free(0);
  }

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
    // resize our cuda frame buffer
    framebuffer.resize(new_size);
    reset_frame();
    // update the launch parameters that we'll pass to the optix launch:
    params.frame.size = framebuffer.size();
    // and re-set the camera, since aspect may have changed
    set_camera(camera_latest);

    // resize auxiliary frame buffers
    framebuffer_accumulation.resize(params.frame.size.long_product() * sizeof(vec4f), framebuffer_stream);
#if defined(ENABLE_OPTIX)
    denoiser.resize(optix_context, framebuffer_stream, new_size);
#endif
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
    volume.set_transfer_function(framebuffer_stream, c, o, r);
    reset_frame();
  }

  void set_volume_sampling_rate(float r)
  {
    volume.set_sampling_rate(r);
    reset_frame();
  }

  void set_volume_density_scale(float s)
  {
    volume.set_density_scale(s);
    reset_frame();
  }

  void set_rendering_mode(int b)
  {
    rendering_mode = b;
    reset_frame();

    program_raymarching.clear(framebuffer_stream);
    program_pathtracing.clear(framebuffer_stream);
  }

  void set_denoiser(bool enable)
  {
#if defined(ENABLE_OPTIX)
    denoiser_enabled = enable;
#endif
  }

  void set_output_as_cuda_framebuffer() { framebuffer_skip_download = true; }

  const StructuredRegularVolume& get_volume() const { return volume; }
  StructuredRegularVolume&       get_volume()       { return volume; }

  void reset_frame() { framebuffer_reset = true; }


  // ------------------------------------------------------------------
  // internal helper functions
  // ------------------------------------------------------------------
protected:
  /*! helper function that initializes optix and checks for errors */
  void initCuda();

#if defined(ENABLE_OPTIX)
  /*! creates and configures a optix device context (in this simple example, only for the primary GPU device) */
  void initOptix();
  /*! build the bottom level acceleration structures */
  void createBLAS();
#endif

  /*! render volume */
  void render_normal();
  void render_neural();

protected:
  /*! @{ CUDA device context and stream that optix pipeline will run on, as well as device properties for this device */
  cudaDeviceProp cuda_device_props{};
  CUcontext cuda_context{};
  cudaStream_t optix_default_stream{};
#if defined(ENABLE_OPTIX)
  OptixDeviceContext optix_context{}; /* the optix context that our pipeline will run in. */
#endif
  /*! @} */

  /*! @{ our launch parameters, on the host, and the buffer to store them on the device */
  LaunchParams params;
  /*! @} */

  NeuralVolume* neural_volume_representation{ nullptr };

#if defined(ENABLE_OPTIX)
  MethodOptiX program_optix;
#endif
  MethodRayMarching program_raymarching;
  MethodPathTracing program_pathtracing;

  // --------------------------------------------------------------- //
  // --------------------------------------------------------------- //
  int rendering_mode{ VNR_INVALID };

  /*! we handle one volume and multiple geometries potentially */
  const cudaTextureObject_t* p_volume_data_texture{nullptr};
  StructuredRegularVolume volume;

#if defined(ENABLE_OPTIX)
  OptixProgram::InstanceHandler volume_instance;                 /*! the ISA handlers */
  std::vector<OptixProgram::InstanceHandler> geometry_instances; /*! the ISA handlers */
#endif

  /*! the rendered image */
  FrameBuffer framebuffer;
  cudaStream_t framebuffer_stream{};
  bool framebuffer_reset{ true };
  CUDABuffer framebuffer_accumulation;
  bool framebuffer_skip_download{ false };

  /*! the camera we are to render with. */
  Camera camera_latest;

#if defined(ENABLE_OPTIX)
  bool denoiser_enabled = false;
  OptixProgramDenoiser denoiser;
#endif
};

} // namespace vnr
