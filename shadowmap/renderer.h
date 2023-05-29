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

#include "device.h"

#include "api_internal.h"
#include "core/framebuffer.h"
#include "core/object.h"
#include "core/sampler.h"

#include "method_raymarching.h"

#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <map>
#include <vector>

namespace ovr::nnvolume {

using vnr::LaunchParams;
using vnr::FrameBuffer;
using vnr::MacroCell;

struct DeviceNNVolume::Impl {
  DeviceNNVolume* parent{ nullptr };

public:
  ~Impl() { framebuffer_accumulation.free(0); }

  void init(int argc, const char** argv, DeviceNNVolume* parent);
  void swap();
  void commit();
  void render();

  void mapframe(FrameBufferData* fb)
  {
    CUDA_CHECK(cudaStreamSynchronize(framebuffer_stream));
    const size_t num_bytes = framebuffer.size().long_product();
    fb->rgba->set_data(framebuffer.device_pointer(), num_bytes * sizeof(vec4f), CrossDeviceBuffer::DEVICE_CUDA);
  }

  void set_shadow(vnrVolume);
  void set_occlusion(vnrVolume);

  void set_scene_clipbox(const box3f& clip) 
  { 
    volume.set_clipping(clip.lower, clip.upper); 
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
  }

  /*! set camera to render with */
  // void set_camera(vec3f from, vec3f at, vec3f up) { set_camera(Camera{ from, at, up }); }
  void set_camera(const Camera& camera) { camera_latest = camera; reset_frame(); }

  void set_transfer_function(const std::vector<vec3f>& c, const std::vector<vec2f>& o, const range1f& r) 
  { 
    volume.set_transfer_function(framebuffer_stream, c, o, r); 
    transfer_function_updated = true;
    reset_frame();
  }

  void set_volume_sampling_rate(float r) { volume.set_sampling_rate(r); reset_frame(); } 
  void set_volume_density_scale(float s) { volume.set_density_scale(s); reset_frame(); }

  void reset_frame() { framebuffer_reset = true; }

  // ------------------------------------------------------------------
  // internal helper functions
  // ------------------------------------------------------------------
protected:
  /*! helper function that initializes optix and checks for errors */
  void initCuda();

  /*! render volume */
  // void render_normal();
  // void render_neural();

protected:
  /*! @{ CUDA device context and stream that optix pipeline will run on, as well as device properties for this device */
  cudaDeviceProp cuda_device_props{};
  CUcontext      cuda_context{};
  cudaStream_t   default_stream{};

  /*! @{ our launch parameters, on the host, and the buffer to store them on the device */
  LaunchParams params;
  /*! @} */

  vnrVolume v_shadow;
  vnrVolume v_occlusion;
  cudaTextureObject_t* simple_shadow{ nullptr };
  cudaTextureObject_t* simple_occlusion{ nullptr };
  NeuralVolume* neural_shadow{ nullptr };
  NeuralVolume* neural_occlusion{ nullptr };

  MethodRayMarching program;

  StructuredRegularVolume volume;
  MacroCell macrocell;
  bool shading = false;

  /*! the rendered image */
  FrameBuffer  framebuffer;
  cudaStream_t framebuffer_stream{};
  CUDABuffer   framebuffer_accumulation;
  bool         framebuffer_reset{ true };

  bool transfer_function_updated{ true };

  /*! the camera we are to render with. */
  Camera camera_latest;
};

}
