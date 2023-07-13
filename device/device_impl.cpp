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
  // // sanity check: make sure we launch only after first resize is already done:
  // if (params.frame.size.x <= 0 || params.frame.size.y <= 0) return;

  // /* framebuffer ... */
  // framebuffer_stream  = framebuffer.current_stream();
  // params.accumulation = (vec4f*)framebuffer_accumulation.d_pointer();
  // params.frame.rgba   = framebuffer.device_pointer();

  // /* volume ... */
  // volume.commit(framebuffer_stream);
  // if (volume.empty()) return;
  // if (transfer_function_updated) {
  //   macrocell.update_max_opacity(volume.device().tfn, 0);
  //   transfer_function_updated = false;
  // }
  // params.transform = volume.matrix;

  // /* camera ... */
  // const Camera& camera = camera_latest;
  // /* the factor '2.f' here might be unnecessary, but I want to match ospray's implementation */
  // const float fovy = camera.perspective.fovy;
  // const float t = 2.f /* (note above) */ * tan(fovy * 0.5f * (float)M_PI / 180.f);
  // const float aspect = params.frame.size.x / float(params.frame.size.y);
  // params.last_camera = params.camera;
  // params.camera.position = camera.from;
  // params.camera.direction = normalize(camera.at - camera.from);
  // params.camera.horizontal = t * aspect * normalize(cross(params.camera.direction, camera.up));
  // params.camera.vertical = cross(params.camera.horizontal, params.camera.direction) / aspect;

  // // /* correct light direction */
  // // if (dot(params.camera.direction, params.light_directional_dir) > 0) {
  // //   params.light_directional_dir *= -1;
  // // }

  // /* reset framebuffer */
  // if (framebuffer_reset) { params.frame_index = 0; }
  // framebuffer_reset = false;
  // params.frame_index++;

  // // float r = 1;
  // // float phi   = 1.0 * M_PI;
  // // float theta = 0.4 * M_PI;
  // // float x = r * cos(phi) * sin(theta);
  // // float y = r * sin(phi) * sin(theta);
  // // float z = r * cos(theta);
  // // params.light_directional_dir.x = x;
  // // params.light_directional_dir.y = y;
  // // params.light_directional_dir.z = z;

  // // it doesnot make sense to render ground truth without a volume texture, so we skip it all together
  // if (shading)
  //   program.render(framebuffer_stream, params, volume, MethodShadowMap::SHADING, neural_occlusion, true);
  // else
  //   program.render(framebuffer_stream, params, volume, MethodShadowMap::NO_SHADING, nullptr, false);

  // framebuffer.download_async();

  renderer.render();
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
  assert(scene.instances.size() == 1 && "[nncache] only accept one instance");
  assert(scene.instances[0].models.size() == 1 && "[nncache] only accept one model");
  assert(scene.instances[0].models[0].type == scene::Model::VOLUMETRIC_MODEL && "[nncache] only accept volume");
  assert(scene.instances[0].models[0].volume_model.volume.type == scene::Volume::STRUCTURED_REGULAR_VOLUME && "[nncache] only accept structured regular volume");
  auto& st = scene.instances[0].models[0].volume_model.transfer_function;
  auto& sv = scene.instances[0].models[0].volume_model.volume.structured_regular;

  // --------------------------------------------
  // create volume texture
  // --------------------------------------------
  Array3DScalarCUDA output = CreateArray3DScalarCUDA(sv.data);
  std::cout << "[nncache] volume range = " << output.lower.v << " " << output.upper.v << std::endl;

  // --------------------------------------------
  // create volume transformation
  // --------------------------------------------
  vec3f scale = sv.grid_spacing * vec3f(sv.data->dims);
  vec3f translate = sv.grid_origin;
  auto matrix = affine3f::translate(translate) * affine3f::scale(scale);

  // --------------------------------------------
  // set macrcell
  // --------------------------------------------
  macrocell.set_shape(output.dims);
  macrocell.allocate();
  macrocell.compute_everything(output.data);

  // // --------------------------------------------
  // // set volume
  // // --------------------------------------------
  // volume.matrix = affine3f::translate(translate) * affine3f::scale(scale);
  // volume.set_volume(output.data, (vnr::ValueType)output.type, output.dims, range1f(output.lower.v, output.upper.v));
  // volume.set_sampling_rate(scene.volume_sampling_rate);
  // volume.set_macrocell(macrocell.dims(), macrocell.spacings(), macrocell.d_value_range(), macrocell.d_max_opacity());
  // volume.get_sbt_pointer(0); // hmm this is necessary

  // --------------------------------------------
  // convert transfer function
  // --------------------------------------------
  std::vector<vec3f> colors_data;
  std::vector<vec2f> alphas_data;
  colors_data.resize(st.color->size());
  for (int i = 0; i < colors_data.size(); ++i) {
    colors_data[i].x = st.color->data_typed<vec4f>()[i].x;
    colors_data[i].y = st.color->data_typed<vec4f>()[i].y;
    colors_data[i].z = st.color->data_typed<vec4f>()[i].z;
  }
  alphas_data.resize(st.opacity->size());
  for (int i = 0; i < alphas_data.size(); ++i) {
    alphas_data[i].x = (float)i / (alphas_data.size() - 1);
    alphas_data[i].y = st.opacity->data_typed<float>()[i];
  }
  set_transfer_function(colors_data, alphas_data, range1f(st.value_range.x, st.value_range.y));

  // vnrJson params = vnrCreateJsonBinary("/home/qadwu/Work/ovr/data/params.json");
  // params.erase("macrocell");
  // params.erase("volume");
  // vnrVolume occlusion = vnrCreateNeuralVolume(params["model"], volume.get_dims());
  // vnrNeuralVolumeSetParams(occlusion, params);
  // set_occlusion(occlusion);

  // this->params.light_directional_dir = scene.lights[0].directional.direction;

  // --------------------------------------------
  //
  // --------------------------------------------
  renderer.set_scene(output.data, 
                     (vnr::ValueType)output.type, 
                     output.dims,
                     range1f(output.lower.v, output.upper.v), 
                     matrix, 
                     macrocell.dims(),
                     macrocell.spacings(),
                     macrocell.d_value_range(),
                     macrocell.d_max_opacity());

  renderer.set_rendering_mode(5);
  renderer.set_output_as_cuda_framebuffer();
  renderer.init();
}

// void 
// DeviceNNVolume::Impl::set_shadow(vnrVolume v) 
// {
//   v_shadow = v;
//   if (v_shadow->isNetwork()) {
//     neural_shadow = &(std::dynamic_pointer_cast<vnr::NeuralVolumeContext>(v_shadow)->neural);
//   }
// }

// void 
// DeviceNNVolume::Impl::set_occlusion(vnrVolume v) 
// {
//   v_occlusion = v;
//   if (v_occlusion->isNetwork()) {
//     neural_occlusion = &(std::dynamic_pointer_cast<vnr::NeuralVolumeContext>(v_occlusion)->neural);
//   }
// }

// void 
// DeviceNNVolume::Impl::swap()
// {
//   // renderer.swap();
// }

void 
DeviceNNVolume::Impl::commit()
{
  if (parent->params.fbsize.update()) {
    resize(parent->params.fbsize.ref());
  }

  /* commit other data */
  if (parent->params.camera.update()) {
    set_camera(parent->params.camera.ref());
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
    set_transfer_function(tfn_colors_data, tfn_alphas_data, range1f(tfn.tfn_value_range.x, tfn.tfn_value_range.y));
  }

  if (parent->params.path_tracing.update()) {
    shading = parent->params.path_tracing.get();
  }

  if (parent->params.volume_sampling_rate.update()) {
    set_volume_sampling_rate(parent->params.volume_sampling_rate.get());
  }
}

} // namespace ovr
