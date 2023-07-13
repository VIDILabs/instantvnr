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
    vec4f *pixels = nullptr; renderer.mapframe(&pixels);
    fb->rgba->set_data(pixels, fbsize.long_product() * sizeof(vec4f), CrossDeviceBuffer::DEVICE_CUDA);
  }

  void set_scene_clipbox(const box3f& clip) { 
    renderer.set_scene_clipbox(clip);
  }

  void resize(const vec2i& size) {
    renderer.resize(size); fbsize = size;
  }

  void set_camera(const Camera& camera) { 
    renderer.set_camera(vnr::Camera{ camera.from, camera.at, camera.up });
  }

  void set_transfer_function(const std::vector<vec3f>& c, const std::vector<vec2f>& o, const range1f& r) { 
    renderer.set_transfer_function(c, o, r);
    macrocell.update_max_opacity(renderer.get_volume().device().tfn, nullptr);
  }

  void set_volume_sampling_rate(float r) { 
    renderer.set_volume_sampling_rate(r); 
  } 

  void set_volume_density_scale(float s) { 
    renderer.set_volume_density_scale(s); 
  }

protected:
  vnr::MainRenderer renderer;
  vnr::MacroCell macrocell;

  // vnrVolume v_occlusion;
  // cudaTextureObject_t* simple_occlusion{ nullptr };
  // vnr::NeuralVolume* neural_occlusion{ nullptr };

  bool shading = false;

  vec2i fbsize;
};

}
