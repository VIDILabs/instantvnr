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
//  device_impl.h
//
//  Pimpl implementation of `DeviceNNVolume`. `Impl::init` reads the OVR
//  scene from `parent->current_scene`, uploads the structured-regular
//  volume through `device_nnvolume_array.h` (`CreateArray3DScalarCUDA`),
//  builds a `MacroCell` + `TransferFunctionAPI`, and initializes the
//  owned `vnr::RenderAPI`. `Impl::commit` pulls updates from the OVR host
//  (`fbsize`, camera, TFN, path-tracing toggle, sampling rate, density
//  scale) via their transactional wrappers and reconfigures the render
//  context accordingly.
// ----------------------------------------------------------------------------

#pragma once

#include "device.h"
#include "device_nnvolume_array.h"

#include "api_internal.h"

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
  ~Impl() {}

  void init(int argc, const char** argv, DeviceNNVolume* parent);
  void commit();
  void render();

  void mapframe(FrameBufferData* fb) {
    CUDA_CHECK(cudaStreamSynchronize(framebuffer_stream));
    vec4f *pixels = framebuffer.device_pointer(); 
    fb->rgba->set_data(pixels, framebuffer_size.long_product() * sizeof(vec4f), 
                       CrossDeviceBuffer::DEVICE_CUDA);

    framebuffer.safe_swap();
    ctx.stream = framebuffer_stream = framebuffer.current_stream();
  }

  void set_scene_clipbox(const box3f& clip) { 
    clipbox = clip;
    framebuffer_reset = true;
  }

protected:

  // --------------------------------------------------------------- //
  int rendering_mode{ 5 };

  // NeuralVolume* neural_volume_representation{ nullptr };
  // /*! we handle one volume and multiple geometries potentially */
  // const cudaTextureObject_t* p_volume_data_texture{nullptr};
  // cudaTextureObject_t volume_data_texture{ 0 };
  // // StructuredRegularVolume volume;

  // --------------------------------------------------------------- //
  /*! the rendered image */
  FrameBuffer framebuffer;
  cudaStream_t framebuffer_stream{};
  bool framebuffer_reset{ true };
  vec2i framebuffer_size;

  // --------------------------------------------------------------- //
  // vnrVolume v_occlusion;
  // cudaTextureObject_t* simple_occlusion{ nullptr };
  // vnr::NeuralVolume* neural_occlusion{ nullptr };

  // volume states
  Array3DScalarCUDA volume_data;
  float sampling_rate{ 1.f };
  float density_scale{ 1.f };
  box3f clipbox = box3f(vec3f(0), vec3f(1)); // object space box
  range1f original_data_range;

  // --------------------------------------------------------------- //
  // handlers
  vnr::Camera camera_latest;
  vnr::MacroCell macrocell;
  vnr::TransferFunctionAPI transfer_function;
  vnr::RenderAPI ctx;
};

}
