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

#include "device_impl.h"
#include "device_nnvolume_array.h"

#include <iostream>

namespace ovr::nnvolume {

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

/*! render one frame */
void
DeviceNNVolume::Impl::render()
{
  // sanity check: make sure we launch only after first resize is already done:
  if (framebuffer_size.x <= 0 || framebuffer_size.y <= 0) return;

  if (framebuffer_reset) {
    ctx.update(
      rendering_mode, 
      transfer_function.tfn,
      sampling_rate,
      density_scale,
      clipbox.lower,
      clipbox.upper,
      camera_latest,
      framebuffer_size
    );
    framebuffer_reset = false;
  }

  ctx.render(framebuffer.device_pointer(), nullptr, volume_data.data);
}

void 
DeviceNNVolume::Impl::init(int argc, const char** argv, DeviceNNVolume* p)
{
  if (parent) {
    throw std::runtime_error("[nncache] device already initialized!");
  }
  parent = p;

  // --------------------------------------------
  // setup scene
  // --------------------------------------------
  const auto& scene = parent->current_scene;
  auto& sv = ovr::parse_single_volume_scene(scene, scene::Volume::STRUCTURED_REGULAR_VOLUME).structured_regular;
  auto& st = scene.instances[0].models[0].volume_model.transfer_function;

  // --------------------------------------------
  // create volume texture
  // create volume transformation
  // --------------------------------------------
  vec3f scale = sv.grid_spacing * vec3f(sv.data->dims);
  vec3f translate = sv.grid_origin;
  affine3f transform = affine3f::translate(translate) * affine3f::scale(scale);
  volume_data = CreateArray3DScalarCUDA(sv.data);
  std::cout << "[nncache] volume range = " << volume_data.lower.v << " " << volume_data.upper.v << std::endl;

  // --------------------------------------------
  // set macrcell
  // --------------------------------------------
  macrocell.set_shape(volume_data.dims);
  macrocell.allocate();
  macrocell.compute_everything(volume_data.data);

  // --------------------------------------------
  // convert transfer function
  // --------------------------------------------
  // std::vector<vec3f> colors_data;
  // std::vector<vec2f> alphas_data;
  // colors_data.resize(st.color->size());
  // for (int i = 0; i < colors_data.size(); ++i) {
  //   colors_data[i].x = st.color->data_typed<vec4f>()[i].x;
  //   colors_data[i].y = st.color->data_typed<vec4f>()[i].y;
  //   colors_data[i].z = st.color->data_typed<vec4f>()[i].z;
  // }
  // alphas_data.resize(st.opacity->size());
  // for (int i = 0; i < alphas_data.size(); ++i) {
  //   alphas_data[i].x = (float)i / (alphas_data.size() - 1);
  //   alphas_data[i].y = st.opacity->data_typed<float>()[i];
  // }
  // set_transfer_function(colors_data, alphas_data, range1f(st.value_range.x, st.value_range.y));

  // vnrJson params = vnrCreateJsonBinary("/home/qadwu/Work/ovr/data/params.json");
  // params.erase("macrocell");
  // params.erase("volume");
  // vnrVolume occlusion = vnrCreateNeuralVolume(params["model"], volume.get_dims());
  // vnrNeuralVolumeSetParams(occlusion, params);
  // set_occlusion(occlusion);

  // this->params.light_directional_dir = scene.lights[0].directional.direction;

  // --------------------------------------------
  // framebuffer creation and initialization
  // --------------------------------------------
  framebuffer.create();
  ctx.stream = framebuffer_stream = framebuffer.current_stream();

  ctx.init(transform,
    (vnr::ValueType)volume_data.type, 
    volume_data.dims, 
    vnr::range1f(volume_data.lower.v, volume_data.upper.v),
    macrocell.dims(),
    macrocell.spacings(),
    macrocell.d_value_range(),
    macrocell.d_max_opacity()
  );

  framebuffer_reset = true;
}

void 
DeviceNNVolume::Impl::commit()
{
  if (parent->params.fbsize.update()) {
    framebuffer_size = parent->params.fbsize.ref();
    framebuffer.resize(framebuffer_size);
    framebuffer_reset = true;
  }

  /* commit other data */
  if (parent->params.camera.update()) {
    const auto& camera = parent->params.camera.ref();
    camera_latest = vnr::Camera{ camera.from, camera.at, camera.up };
    framebuffer_reset = true;
  }

  if (parent->params.tfn.update()) {
    const auto& tfn = parent->params.tfn.ref();
    std::vector<vec3f> tfn_colors_data;
    std::vector<vec2f> tfn_alphas_data;
    tfn_colors_data.resize(tfn.tfn_colors.size() / 3);
    for (int i = 0; i < tfn_colors_data.size(); ++i) {
      tfn_colors_data[i].x = tfn.tfn_colors[3 * i + 0];
      tfn_colors_data[i].y = tfn.tfn_colors[3 * i + 1];
      tfn_colors_data[i].z = tfn.tfn_colors[3 * i + 2];
    }
    tfn_alphas_data.resize(tfn.tfn_alphas.size() / 2);
    for (int i = 0; i < tfn_alphas_data.size(); ++i) {
      tfn_alphas_data[i].x = tfn.tfn_alphas[2 * i + 0];
      tfn_alphas_data[i].y = tfn.tfn_alphas[2 * i + 1];
    }

    vnr::TransferFunction data;
    data.color = tfn_colors_data;
    data.alpha = tfn_alphas_data;
    data.range = range1f(tfn.tfn_value_range.x, tfn.tfn_value_range.y);

    vnr::range1f range = vnr::range1f(volume_data.lower.v, volume_data.upper.v);
  
    transfer_function.update(data, range, ctx.stream);

    macrocell.update_max_opacity(transfer_function.tfn, ctx.stream);
    framebuffer_reset = true;
  }

  if (parent->params.path_tracing.update()) {
    if (parent->params.path_tracing.get()) {
      rendering_mode =  VNR_PATHTRACING_SAMPLE_STREAMING;
    }
    else {
      rendering_mode =  VNR_RAYMARCHING_NO_SHADING_SAMPLE_STREAMING;
    }
    framebuffer_reset = true;
  }

  if (parent->params.volume_sampling_rate.update()) {
    sampling_rate = parent->params.volume_sampling_rate.get();
    framebuffer_reset = true;
  }

  if (parent->params.volume_density_scale.update()) {
    density_scale = parent->params.volume_density_scale.get();
    framebuffer_reset = true;
  }
}

} // namespace ovr
