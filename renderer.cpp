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

static void
context_log_cb(unsigned int level, const char* tag, const char* message, void*)
{
  // fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

static void
general_log_cb(const char* log, size_t sizeof_log)
{
  // if (sizeof_log > 1) PRINT(log);
}

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

/*! render one frame */
void
MainRenderer::render()
{
  // // sanity check: make sure we launch only after first resize is already done:
  // if (params.frame.size.x <= 0 || params.frame.size.y <= 0) return;

  /* framebuffer ... */
  framebuffer_stream = framebuffer.current_stream();
  // params.accumulation = (vec4f*)framebuffer_accumulation.d_pointer();
  // params.frame.rgba = framebuffer.device_pointer();

  // /* volumes ... */
  // // volume texture might get updated during rendering, we will keep an eye on it.
  // if (volume.empty() && *p_volume_data_texture != 0) {
  //   volume.set_volume(*p_volume_data_texture);
  // }
  // volume.commit(framebuffer_stream);
  // params.transform = volume.matrix;

  // /* camera ... */
  // const Camera& camera = camera_latest;
  // /* the factor '2.f' here might be unnecessary, but I want to match ospray's implementation */
  // const float fovy = camera.fovy;
  // const float t = 2.f /* (note above) */ * tan(fovy * 0.5f * (float)M_PI / 180.f);
  // const float aspect = params.frame.size.x / float(params.frame.size.y);
  // params.last_camera = params.camera;
  // params.camera.position = camera.from;
  // params.camera.direction = normalize(camera.at - camera.from);
  // params.camera.horizontal = t * aspect * normalize(cross(params.camera.direction, camera.up));
  // params.camera.vertical = cross(params.camera.horizontal, params.camera.direction) / aspect;

  // /* correct light direction */
  // if (dot(params.camera.direction, params.light_directional_dir) > 0) {
  //   params.light_directional_dir *= -1;
  // }

  // /* reset framebuffer */
  // if (framebuffer_reset) { params.frame_index = 0; }
  // params.frame_index++;
  // // params.frame_index = 1;

  if (volume_data_texture == 0 && *p_volume_data_texture != 0) {
    volume_data_texture = (*p_volume_data_texture);
  }

  // ------------------------------------- //
  ctx.stream = framebuffer_stream;
  if (framebuffer_reset) {
    ctx.update(rendering_mode, 
      tfn,
      sampling_rate,
      density_scale,
      bbox.lower,
      bbox.upper,
      camera_latest,
      framebuffer_size
    );
  }
  ctx.render(framebuffer.device_pointer(), neural_volume_representation, volume_data_texture);
  // ------------------------------------- //

  // /* draw call */
  // if (!neural_volume_representation) {
  //   render_normal();
  // }
  // else {
  //   render_neural();
  // }

  // // finalize frame
  // try {
  //   // CUDA_SYNC_CHECK();
  // }
  // catch (std::runtime_error& e) {
  //   std::cerr << e.what() << std::endl;
  //   return;
  // }

  framebuffer_reset = false;
  if (!framebuffer_skip_download) framebuffer.download_async();

  // sync - make sure the frame is rendered before we download and
  // display (obviously, for a high-performance application you
  // want to use streams and double-buffering, but for this simple
  // example, this will have to do)
  // CUDA_SYNC_CHECK();
}

// void
// MainRenderer::render_normal()
// {
//   // it doesnot make sense to render ground truth without a volume texture, so
//   // we skip it all together
//   if (volume.empty()) return;

//   assert(!neural_volume_representation);

//   /* rendering */
//   switch (rendering_mode) {
//   // path tracing
//   case VNR_PATHTRACING_DECODING:         program_pathtracing.render(framebuffer_stream, params, volume.d_pointer());                 break;
//   case VNR_PATHTRACING_SAMPLE_STREAMING: program_pathtracing.render(framebuffer_stream, params, volume.d_pointer(), nullptr, true);  break;
//   case VNR_PATHTRACING_IN_SHADER:        program_pathtracing.render(framebuffer_stream, params, volume.d_pointer(), nullptr, false); break;
//   // ray marching
//   case VNR_RAYMARCHING_NO_SHADING_DECODING:         program_raymarching.render(framebuffer_stream, params, MethodRayMarching::NO_SHADING, volume.d_pointer());                 break;
//   case VNR_RAYMARCHING_NO_SHADING_SAMPLE_STREAMING: program_raymarching.render(framebuffer_stream, params, MethodRayMarching::NO_SHADING, volume.d_pointer(), nullptr, true);  break;
//   case VNR_RAYMARCHING_NO_SHADING_IN_SHADER:        program_raymarching.render(framebuffer_stream, params, MethodRayMarching::NO_SHADING, volume.d_pointer(), nullptr, false); break;
//   // ray marching
//   case VNR_RAYMARCHING_GRADIENT_SHADING_DECODING:         program_raymarching.render(framebuffer_stream, params, MethodRayMarching::GRADIENT_SHADING, volume.d_pointer());                 break;
//   case VNR_RAYMARCHING_GRADIENT_SHADING_SAMPLE_STREAMING: program_raymarching.render(framebuffer_stream, params, MethodRayMarching::GRADIENT_SHADING, volume.d_pointer(), nullptr, true);  break;
//   case VNR_RAYMARCHING_GRADIENT_SHADING_IN_SHADER:        program_raymarching.render(framebuffer_stream, params, MethodRayMarching::GRADIENT_SHADING, volume.d_pointer(), nullptr, false); break;
//   // ray marching
//   case VNR_RAYMARCHING_SINGLE_SHADE_HEURISTIC_DECODING:         program_raymarching.render(framebuffer_stream, params, MethodRayMarching::SINGLE_SHADE_HEURISTIC, volume.d_pointer());                 break;
//   case VNR_RAYMARCHING_SINGLE_SHADE_HEURISTIC_SAMPLE_STREAMING: program_raymarching.render(framebuffer_stream, params, MethodRayMarching::SINGLE_SHADE_HEURISTIC, volume.d_pointer(), nullptr, true);  break;
//   case VNR_RAYMARCHING_SINGLE_SHADE_HEURISTIC_IN_SHADER:        program_raymarching.render(framebuffer_stream, params, MethodRayMarching::SINGLE_SHADE_HEURISTIC, volume.d_pointer(), nullptr, false); break;
//   default: break;
//   }
// }

// void
// MainRenderer::render_neural()
// {
//   assert(neural_volume_representation);
//   auto* nvr = neural_volume_representation;

//   /* rendering */
//   switch (rendering_mode) {
//   // path tracing
//   case VNR_PATHTRACING_DECODING:         program_pathtracing.render(framebuffer_stream, params, volume.d_pointer());             break;
//   case VNR_PATHTRACING_SAMPLE_STREAMING: program_pathtracing.render(framebuffer_stream, params, volume.d_pointer(), nvr, true);  break;
//   case VNR_PATHTRACING_IN_SHADER:        program_pathtracing.render(framebuffer_stream, params, volume.d_pointer(), nvr, false); break;
//   // ray marching
//   case VNR_RAYMARCHING_NO_SHADING_DECODING:         program_raymarching.render(framebuffer_stream, params, MethodRayMarching::NO_SHADING, volume.d_pointer());             break;
//   case VNR_RAYMARCHING_NO_SHADING_SAMPLE_STREAMING: program_raymarching.render(framebuffer_stream, params, MethodRayMarching::NO_SHADING, volume.d_pointer(), nvr, true);  break;
//   case VNR_RAYMARCHING_NO_SHADING_IN_SHADER:        program_raymarching.render(framebuffer_stream, params, MethodRayMarching::NO_SHADING, volume.d_pointer(), nvr, false); break;
//   // ray marching
//   case VNR_RAYMARCHING_GRADIENT_SHADING_DECODING:         program_raymarching.render(framebuffer_stream, params, MethodRayMarching::GRADIENT_SHADING, volume.d_pointer());             break;
//   case VNR_RAYMARCHING_GRADIENT_SHADING_SAMPLE_STREAMING: program_raymarching.render(framebuffer_stream, params, MethodRayMarching::GRADIENT_SHADING, volume.d_pointer(), nvr, true);  break;
//   case VNR_RAYMARCHING_GRADIENT_SHADING_IN_SHADER:        program_raymarching.render(framebuffer_stream, params, MethodRayMarching::GRADIENT_SHADING, volume.d_pointer(), nvr, false); break;
//   // ray marching
//   case VNR_RAYMARCHING_SINGLE_SHADE_HEURISTIC_DECODING:         program_raymarching.render(framebuffer_stream, params, MethodRayMarching::SINGLE_SHADE_HEURISTIC, volume.d_pointer());             break;
//   case VNR_RAYMARCHING_SINGLE_SHADE_HEURISTIC_SAMPLE_STREAMING: program_raymarching.render(framebuffer_stream, params, MethodRayMarching::SINGLE_SHADE_HEURISTIC, volume.d_pointer(), nvr, true);  break;
//   case VNR_RAYMARCHING_SINGLE_SHADE_HEURISTIC_IN_SHADER:        program_raymarching.render(framebuffer_stream, params, MethodRayMarching::SINGLE_SHADE_HEURISTIC, volume.d_pointer(), nvr, false); break;
//   default: break;
//   }
// }

void
MainRenderer::set_scene(const cudaTextureObject_t& texture, ValueType type, vec3i dims, range1f range, 
                        affine3f transform, vec3i macrocell_dims, vec3f macrocell_spacings, 
                        vec2f* macrocell_d_value_range, float* macrocell_d_max_opacity,
                        NeuralVolume* neural_representation)
{
  neural_volume_representation = neural_representation;

  p_volume_data_texture = &texture;
  volume_data_texture = texture;
  original_data_range = range;

  // printf("[vnr] MacroCell: Dims = (%d,%d,%d) Spacing (%f,%f,%f)\n",
  //        macrocell_dims.x, macrocell_dims.y,macrocell_dims.z,
  //        macrocell_spacings.x,
  //        macrocell_spacings.y,
  //        macrocell_spacings.z);

  // /* create a volume texture regularly */
  // auto& v = volume;
  // v.matrix = transform;
  // v.set_sampling_rate(1.f);
  // v.set_volume(texture, type, dims, range);
  // v.set_macrocell(macrocell_dims, macrocell_spacings, macrocell_d_value_range, macrocell_d_max_opacity);

  /* book keeping (might not be necessary) */
  framebuffer_reset = true;

  // ------------------------------------- //
  ctx.init(transform, 
    type, dims, range,  
    macrocell_dims, 
    macrocell_spacings, 
    macrocell_d_value_range, 
    macrocell_d_max_opacity
  );
  // ------------------------------------- //
}

void
MainRenderer::set_scene_clipbox(const box3f& clip)
{
  // volume.set_clipping(clip.lower, clip.upper);
  bbox = clip;
}

MainRenderer::~MainRenderer()
{
  // framebuffer_accumulation.free(0);

  if (tfn_color_array_handler) {
    CUDA_CHECK_NOEXCEPT(cudaTrackedFreeArray(tfn_color_array_handler));
    tfn_color_array_handler = NULL;
  }
  if (tfn.colors.data) {
    CUDA_CHECK_NOEXCEPT(cudaDestroyTextureObject(tfn.colors.data));
    tfn.colors.data = { 0 };
  }
  if (tfn.colors.rawptr) {
    CUDA_CHECK_NOEXCEPT(cudaTrackedFree(tfn.colors.rawptr, tfn.colors.length * sizeof(float4)));
    tfn.colors.rawptr = nullptr;
  }
  tfn.colors.length = 0;

  if (tfn_alpha_array_handler) {
    CUDA_CHECK_NOEXCEPT(cudaTrackedFreeArray(tfn_alpha_array_handler));
    tfn_color_array_handler = NULL;
  }
  if (tfn.alphas.data) {
    CUDA_CHECK_NOEXCEPT(cudaDestroyTextureObject(tfn.alphas.data));
    tfn.alphas.data = { 0 };
  }
  if (tfn.alphas.rawptr) {
    CUDA_CHECK_NOEXCEPT(cudaTrackedFree(tfn.alphas.rawptr, tfn.alphas.length * sizeof(float)));
    tfn.alphas.rawptr = nullptr;
  }
  tfn.alphas.length = 0;

}

/*! constructor - performs all setup, including initializing
  optix, creates module, pipeline, programs, SBT, etc. */
void
MainRenderer::init()
{
  initCuda();
  framebuffer.create();
  log() << "[vnr] " << GDT_TERMINAL_GREEN;
  log() << "Instant Neural Representation Renderer is Ready" << std::endl;
  log() << GDT_TERMINAL_DEFAULT;
}

/*! helper function that initializes optix and checks for errors */
void
MainRenderer::initCuda()
{
  // -------------------------------------------------------
  // check for available optix7 capable devices
  // -------------------------------------------------------
  cudaFree(0);
  int num_devices; cudaGetDeviceCount(&num_devices);
  if (num_devices == 0) throw std::runtime_error("[vnr] no CUDA capable devices found!");
  log() << "[vnr] found " << num_devices << " CUDA devices" << std::endl;

  // -------------------------------------------------------
  // for this sample, do everything on one device
  // -------------------------------------------------------
  int device_id = 0;
  if (const char* env_p = std::getenv("VNR_CUDA_DEVICE")) {
    device_id = std::stoi(env_p);
    std::cout << "[vnr] VNR_CUDA_DEVICE: " << device_id << std::endl;
  }
  CUDA_CHECK(cudaSetDevice(device_id));

  cudaDeviceProp cuda_device_props{};
  CUcontext cuda_context{};

  // char cuda_pci_bus[32];
  // cudaDeviceGetPCIBusId(cuda_pci_bus, 32, device_id);
  // cudaGetDeviceProperties(&cuda_device_props, device_id);
  // std::cout << "[vnr] running on device: " << cuda_device_props.name << " (" << std::string(cuda_pci_bus) << ")" << std::endl;

  CUresult result = cuCtxGetCurrent(&cuda_context);
  if (result != CUDA_SUCCESS)
    fprintf(stderr, "Error querying current context: error code %d\n", result);
}

} // namespace ovr
