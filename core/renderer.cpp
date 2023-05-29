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

#if defined(ENABLE_OPTIX)
// this include may only appear in a single source file:
#include <optix_function_table_definition.h>
#endif

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
  // sanity check: make sure we launch only after first resize is already done:
  if (params.frame.size.x <= 0 || params.frame.size.y <= 0) return;

  /* framebuffer ... */
  framebuffer_stream = framebuffer.current_stream();
  params.accumulation = (vec4f*)framebuffer_accumulation.d_pointer();
#if defined(ENABLE_OPTIX)
  if (denoiser_enabled) {
    params.frame.rgba = denoiser.d_ptr();
  }
  else
#endif
  {
    params.frame.rgba = framebuffer.device_pointer();
  }

  /* volumes ... */
  // volume texture might get updated during rendering, we will keep an eye on it.
  if (volume.empty() && *p_volume_data_texture != 0) {
    volume.set_volume(*p_volume_data_texture);
  }
  volume.commit(framebuffer_stream);
  params.transform = volume.matrix;

  /* geometries ... */
  // for (auto& b : boxes) b.commit(framebuffer_stream); // not needed for boxes geometry

  /* camera ... */
  const Camera& camera = camera_latest;
  /* the factor '2.f' here might be unnecessary, but I want to match ospray's implementation */
  const float fovy = camera.fovy;
  const float t = 2.f /* (note above) */ * tan(fovy * 0.5f * M_PI / 180.f);
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

  /* reset framebuffer */
  if (framebuffer_reset) { params.frame_index = 0; }
  params.frame_index++;
  // params.frame_index = 1;

  /* draw call */
  if (!neural_volume_representation) {
    render_normal();
  }
  else {
    render_neural();
  }

  // denoise
#if defined(ENABLE_OPTIX)
  if (denoiser_enabled) {
    denoiser.process(optix_context, framebuffer_stream, !framebuffer_reset, params.frame_index, params.frame.size, framebuffer.device_pointer());
  }
#endif

  // // finalize frame
  // try {
  //   // CUDA_SYNC_CHECK();
  // }
  // catch (std::runtime_error& e) {
  //   std::cerr << e.what() << std::endl;
  //   return;
  // }

  framebuffer_reset = false;
  framebuffer.download_async();

  // sync - make sure the frame is rendered before we download and
  // display (obviously, for a high-performance application you
  // want to use streams and double-buffering, but for this simple
  // example, this will have to do)
  // CUDA_SYNC_CHECK();
}

void
MainRenderer::render_normal()
{
  // it doesnot make sense to render ground truth without a volume texture, so
  // we skip it all together
  if (volume.empty()) return;

  assert(!neural_volume_representation);

  /* rendering */
  switch (rendering_mode) {
  // path tracing
  case VNR_PATHTRACING_DECODING:         program_pathtracing.render(framebuffer_stream, params, volume); break;
  case VNR_PATHTRACING_SAMPLE_STREAMING: program_pathtracing.render(framebuffer_stream, params, volume, nullptr, true); break;
  case VNR_PATHTRACING_IN_SHADER:        program_pathtracing.render(framebuffer_stream, params, volume, nullptr, false); break;
  // ray marching
  case VNR_RAYMARCHING_NO_SHADING_DECODING:         program_raymarching.render(framebuffer_stream, params, volume, MethodRayMarching::NO_SHADING); break;
  case VNR_RAYMARCHING_NO_SHADING_SAMPLE_STREAMING: program_raymarching.render(framebuffer_stream, params, volume, MethodRayMarching::NO_SHADING, nullptr, true); break;
  case VNR_RAYMARCHING_NO_SHADING_IN_SHADER:        program_raymarching.render(framebuffer_stream, params, volume, MethodRayMarching::NO_SHADING, nullptr, false); break;
  // ray marching
  case VNR_RAYMARCHING_GRADIENT_SHADING_DECODING:         program_raymarching.render(framebuffer_stream, params, volume, MethodRayMarching::GRADIENT_SHADING); break;
  case VNR_RAYMARCHING_GRADIENT_SHADING_SAMPLE_STREAMING: program_raymarching.render(framebuffer_stream, params, volume, MethodRayMarching::GRADIENT_SHADING, nullptr, true); break;
  case VNR_RAYMARCHING_GRADIENT_SHADING_IN_SHADER:        program_raymarching.render(framebuffer_stream, params, volume, MethodRayMarching::GRADIENT_SHADING, nullptr, false); break;
  // ray marching
  case VNR_RAYMARCHING_SINGLE_SHADE_HEURISTIC_DECODING:         program_raymarching.render(framebuffer_stream, params, volume, MethodRayMarching::SINGLE_SHADE_HEURISTIC); break;
  case VNR_RAYMARCHING_SINGLE_SHADE_HEURISTIC_SAMPLE_STREAMING: program_raymarching.render(framebuffer_stream, params, volume, MethodRayMarching::SINGLE_SHADE_HEURISTIC, nullptr, true); break;
  case VNR_RAYMARCHING_SINGLE_SHADE_HEURISTIC_IN_SHADER:        program_raymarching.render(framebuffer_stream, params, volume, MethodRayMarching::SINGLE_SHADE_HEURISTIC, nullptr, false); break;
  // reference renderer
#if defined(ENABLE_OPTIX)
  case VNR_OPTIX_NO_SHADING:             program_optix.render(framebuffer_stream, params, MethodOptiX::NO_SHADING);             break;
  case VNR_OPTIX_GRADIENT_SHADING:       program_optix.render(framebuffer_stream, params, MethodOptiX::GRADIENT_SHADING);       break;
  case VNR_OPTIX_FULL_SHADOW:            program_optix.render(framebuffer_stream, params, MethodOptiX::FULL_SHADOW);            break;
  case VNR_OPTIX_SINGLE_SHADE_HEURISTIC: program_optix.render(framebuffer_stream, params, MethodOptiX::SINGLE_SHADE_HEURISTIC); break;
#endif
  default: break;
  }
}

void
MainRenderer::render_neural()
{
  assert(neural_volume_representation);
  auto* nvr = neural_volume_representation;

  /* rendering */
  switch (rendering_mode) {
  // path tracing
  case VNR_PATHTRACING_DECODING:         program_pathtracing.render(framebuffer_stream, params, volume); break;
  case VNR_PATHTRACING_SAMPLE_STREAMING: program_pathtracing.render(framebuffer_stream, params, volume, nvr, true); break;
  case VNR_PATHTRACING_IN_SHADER:        program_pathtracing.render(framebuffer_stream, params, volume, nvr, false); break;
  // ray marching
  case VNR_RAYMARCHING_NO_SHADING_DECODING:         program_raymarching.render(framebuffer_stream, params, volume, MethodRayMarching::NO_SHADING); break;
  case VNR_RAYMARCHING_NO_SHADING_SAMPLE_STREAMING: program_raymarching.render(framebuffer_stream, params, volume, MethodRayMarching::NO_SHADING, nvr, true); break;
  case VNR_RAYMARCHING_NO_SHADING_IN_SHADER:        program_raymarching.render(framebuffer_stream, params, volume, MethodRayMarching::NO_SHADING, nvr, false); break;
  // ray marching
  case VNR_RAYMARCHING_GRADIENT_SHADING_DECODING:         program_raymarching.render(framebuffer_stream, params, volume, MethodRayMarching::GRADIENT_SHADING); break;
  case VNR_RAYMARCHING_GRADIENT_SHADING_SAMPLE_STREAMING: program_raymarching.render(framebuffer_stream, params, volume, MethodRayMarching::GRADIENT_SHADING, nvr, true); break;
  case VNR_RAYMARCHING_GRADIENT_SHADING_IN_SHADER:        program_raymarching.render(framebuffer_stream, params, volume, MethodRayMarching::GRADIENT_SHADING, nvr, false); break;
  // ray marching
  case VNR_RAYMARCHING_SINGLE_SHADE_HEURISTIC_DECODING:         program_raymarching.render(framebuffer_stream, params, volume, MethodRayMarching::SINGLE_SHADE_HEURISTIC); break;
  case VNR_RAYMARCHING_SINGLE_SHADE_HEURISTIC_SAMPLE_STREAMING: program_raymarching.render(framebuffer_stream, params, volume, MethodRayMarching::SINGLE_SHADE_HEURISTIC, nvr, true); break;
  case VNR_RAYMARCHING_SINGLE_SHADE_HEURISTIC_IN_SHADER:        program_raymarching.render(framebuffer_stream, params, volume, MethodRayMarching::SINGLE_SHADE_HEURISTIC, nvr, false); break;
  // reference renderer
#if defined(ENABLE_OPTIX)
  case VNR_OPTIX_NO_SHADING:             program_optix.render(framebuffer_stream, params, MethodOptiX::NO_SHADING);             break;
  case VNR_OPTIX_GRADIENT_SHADING:       program_optix.render(framebuffer_stream, params, MethodOptiX::GRADIENT_SHADING);       break;
  case VNR_OPTIX_FULL_SHADOW:            program_optix.render(framebuffer_stream, params, MethodOptiX::FULL_SHADOW);            break;
  case VNR_OPTIX_SINGLE_SHADE_HEURISTIC: program_optix.render(framebuffer_stream, params, MethodOptiX::SINGLE_SHADE_HEURISTIC); break;
#endif
  default: break;
  }
}

void
MainRenderer::set_scene(const cudaTextureObject_t& texture, ValueType type, vec3i dims, range1f range, 
                        affine3f transform, vec3i macrocell_dims, vec3f macrocell_spacings, 
                        vec2f* macrocell_d_value_range, float* macrocell_d_max_opacity,
                        NeuralVolume* neural_representation)
{
  neural_volume_representation = neural_representation;

  p_volume_data_texture = &texture;

  printf("[vnr] MacroCell: Dims = (%d,%d,%d) Spacing (%f,%f,%f)\n",
         macrocell_dims.x, macrocell_dims.y,macrocell_dims.z,
         macrocell_spacings.x,
         macrocell_spacings.y,
         macrocell_spacings.z);

  /* create a volume texture regularly */
  auto& v = volume;
  v.matrix = transform;
  v.set_sampling_rate(1.f);
  v.set_volume(texture, type, dims, range);
  v.set_macrocell(macrocell_dims, macrocell_spacings, macrocell_d_value_range, macrocell_d_max_opacity);

  /* create geometries */
  // const vec3i& block_dims = params.volume_block_dims;
  // const vec3i& block_counts = params.volume_block_counts;
  // BoxesGeometry b;
  // b.matrix = affine3f::translate(translate);
  // for (int bz = 0; bz < block_counts.z; ++bz) {
  //   for (int by = 0; by < block_counts.y; ++by) {
  //     for (int bx = 0; bx < block_counts.x; ++bx) {
  //       b.add_cube(vec3f(block_dims) * (vec3f(bx, by, bz) + 0.5f), 0.5f * vec3f(block_dims));
  //     }
  //   }
  // }
  // boxes.push_back(b);
  // 
  // boxes.emplace_back();
  // boxes.back().add_cube(vec3f(0.f, -dims.y, 0.f), vec3f(10000.f, 1.f, 10000.f));

  /* book keeping (might not be necessary) */
  framebuffer_reset = true;
}

void
MainRenderer::set_scene_clipbox(const box3f& clip)
{
  volume.set_clipping(clip.lower, clip.upper);
}

/*! constructor - performs all setup, including initializing
  optix, creates module, pipeline, programs, SBT, etc. */
void
MainRenderer::init()
{
  initCuda();

#if defined(ENABLE_OPTIX) 
  initOptix();
  createBLAS();
#endif

  framebuffer.create();

  // generate SBT records for 
  std::map<ObjectType, std::vector<void*>> records;
  {
    auto it = records.emplace(VOLUME_STRUCTURED_REGULAR, 1);
    it.first->second[0] = (void*)volume.get_sbt_pointer(optix_default_stream);
  }
  {
    auto it = records.emplace(GEOMETRY_BOXES, boxes.size());
    assert(it.second && "cannot create SBT for boxes geometry");
    for (int i = 0; i < boxes.size(); ++i) {
      it.first->second[i] = (void*)boxes[i].get_sbt_pointer(optix_default_stream);
    }
  }

#if defined(ENABLE_OPTIX)
  // define BLAS groups
  std::vector<std::vector<OptixProgram::InstanceHandler>> blas;
  blas.push_back(std::vector<OptixProgram::InstanceHandler>{ volume_instance });
  blas.push_back(geometry_instances);
  // create optix program
  program_optix.init(optix_context, records, blas);
#endif

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
  int num_devices;
  cudaGetDeviceCount(&num_devices);
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

  cudaGetDeviceProperties(&cuda_device_props, device_id);
  std::cout << "[vnr] running on device: " << cuda_device_props.name << std::endl;

  CUresult result = cuCtxGetCurrent(&cuda_context);
  if (result != CUDA_SUCCESS)
    fprintf(stderr, "Error querying current context: error code %d\n", result);
}

/*! creates and configures a optix device context (in this simple
  example, only for the primary GPU device) */
#if defined(ENABLE_OPTIX)
void
MainRenderer::initOptix()
{
  // CUDA_CHECK(cudaStreamCreate(&optix_default_stream));
  optix_default_stream = 0;

  // -------------------------------------------------------
  // initialize optix
  // -------------------------------------------------------
  OPTIX_CHECK(optixInit());

  OPTIX_CHECK(optixDeviceContextCreate(cuda_context, 0, &optix_context));
  OPTIX_CHECK(optixDeviceContextSetLogCallback(optix_context, context_log_cb, nullptr, 4));
}
#endif

/*! build the bottom level acceleration structures */
#if defined(ENABLE_OPTIX)
void
MainRenderer::createBLAS()
{
  /*! create the volume ISA handler */
  {
    // we want to treat one volume as a instance
    volume_instance.type = VOLUME_STRUCTURED_REGULAR;
    volume_instance.idx = 0;
    volume.transform(volume_instance.handler.transform);
    volume_instance.handler.instanceId = 0;
    volume_instance.handler.visibilityMask = OptixVisibilityMask(VISIBILITY_MASK_VOLUME);
    volume_instance.handler.sbtOffset = 0xFFFFFFFF; /* invalid */
    volume_instance.handler.flags = OPTIX_INSTANCE_FLAG_NONE;
    volume_instance.handler.traversableHandle = volume.buildas(optix_context, /*stream=*/0);
  }

  /*! create geometry ISA handlers */
  for (int i = 0; i < boxes.size(); ++i) {
    OptixProgram::InstanceHandler instance;
    instance.type = GEOMETRY_BOXES;
    instance.idx = i;
    boxes[i].transform(instance.handler.transform);
    instance.handler.instanceId = 1 + i;
    instance.handler.visibilityMask = OptixVisibilityMask(VISIBILITY_MASK_GEOMETRY);
    instance.handler.sbtOffset = 0xFFFFFFFF; /* invalid */
    instance.handler.flags = OPTIX_INSTANCE_FLAG_NONE;
    instance.handler.traversableHandle = boxes[i].buildas(optix_context, /*stream=*/0);

    geometry_instances.push_back(instance);
  }
}
#endif

#if defined(ENABLE_OPTIX)

/*! creates the module that contains all the programs we are going
    to use. in this simple example, we use a single module from a
    single .cu file, using a single embedded ptx string */
void
OptixProgram::createModule(OptixDeviceContext optix_context)
{
  module.compile_opts.maxRegisterCount = 100;

  module.compile_opts.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
  // module.compile_opts.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_2;
  // module.compile_opts.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_1;
  // module.compile_opts.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;

  module.compile_opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;
  // module.compile_opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
  // module.compile_opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
  // module.compile_opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  pipeline.compile_opts = {};
  pipeline.compile_opts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
  pipeline.compile_opts.usesMotionBlur = false;
  pipeline.compile_opts.numPayloadValues = 2;
  pipeline.compile_opts.numAttributeValues = 8;
  pipeline.compile_opts.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  pipeline.compile_opts.pipelineLaunchParamsVariableName = "optixLaunchParams";

  pipeline.link_opts.maxTraceDepth = 16;

  char log[2048];
  size_t sizeof_log = sizeof(log);
  OPTIX_CHECK(optixModuleCreateFromPTX(optix_context,
                                       &module.compile_opts,
                                       &pipeline.compile_opts,
                                       /* shader program */ ptx_code.c_str(),
                                       ptx_code.size(),
                                       /* logs and output */ log,
                                       &sizeof_log,
                                       &module.handle));
  general_log_cb(log, sizeof_log);
}

/*! assembles the full pipeline of all programs */
void
OptixProgram::createPipeline(OptixDeviceContext optix_context)
{
  std::vector<OptixProgramGroup> program_groups;
  for (auto pg : program.raygens)
    program_groups.push_back(pg);
  for (auto pg : program.misses)
    program_groups.push_back(pg);
  for (auto pg : program.hitgroups)
    program_groups.push_back(pg);

  char log[2048];
  size_t sizeof_log = sizeof(log);
  OPTIX_CHECK(optixPipelineCreate(optix_context,
                                  &pipeline.compile_opts,
                                  &pipeline.link_opts,
                                  program_groups.data(),
                                  (int)program_groups.size(),
                                  log,
                                  &sizeof_log,
                                  &pipeline.handle));
  general_log_cb(log, sizeof_log);

  OPTIX_CHECK(
    optixPipelineSetStackSize(/* [in] The pipeline to configure the stack size for */
                              pipeline.handle,
                              /* [in] The direct stack size requirement for direct callables invoked from IS or AH. */
                              2 * 1024,
                              /* [in] The direct stack size requirement for direct callables invoked from RG, MS, or CH. */
                              2 * 1024,
                              /* [in] The continuation stack requirement. */
                              2 * 1024,
                              /* [in] The maximum depth of a traversable graph passed to trace. */
                              3));
}

/*! does all setup for the raygen program(s) we are going to use */
void
OptixProgram::createRaygenPrograms(OptixDeviceContext optix_context)
{
  // we do a single ray gen program in this example:
  program.raygens.resize(1);

  OptixProgramGroupOptions options = {};
  OptixProgramGroupDesc desc = {};
  desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  desc.raygen.module = module.handle;
  desc.raygen.entryFunctionName = shader_raygen.c_str();

  char log[2048];
  size_t sizeof_log = sizeof(log);
  OPTIX_CHECK(optixProgramGroupCreate(optix_context, &desc, 1, &options, log, &sizeof_log, &program.raygens[0]));
  general_log_cb(log, sizeof_log);
}

/*! does all setup for the miss program(s) we are going to use */
void
OptixProgram::createMissPrograms(OptixDeviceContext optix_context)
{
  // we do a single ray gen program in this example:
  program.misses.resize(shader_misses.size());

  OptixProgramGroupOptions options = {};
  OptixProgramGroupDesc desc = {};
  char log[2048];
  size_t sizeof_log = sizeof(log);

  for (int i = 0; i < shader_misses.size(); ++i) {
    sizeof_log = sizeof(log);
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    desc.miss.module = module.handle;
    desc.miss.entryFunctionName = shader_misses[i].c_str();
    OPTIX_CHECK(optixProgramGroupCreate(optix_context, &desc, 1, &options, log, &sizeof_log, &program.misses[i]));
    general_log_cb(log, sizeof_log);
  }
}

/*! does all setup for the hitgroup program(s) we are going to use */
void
OptixProgram::createHitgroupPrograms(OptixDeviceContext optix_context)
{
  OptixProgramGroupDesc desc = {};
  OptixProgramGroupOptions options = {};
  memset(&options, 0, sizeof(OptixProgramGroupOptions));

  // for this simple example, we set up a single hit group
  char log[2048];
  size_t sizeof_log;

  // cleanup hitgroup
  program.hitgroups.clear();

  // create hitgroup records
  for (auto& shaders : shader_objects) {

    for (auto& s : shaders.hitgroup) {
      memset(&desc, 0, sizeof(OptixProgramGroupDesc));
      desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

      if (!s.shader_CH.empty()) {
        desc.hitgroup.moduleCH = module.handle;
        desc.hitgroup.entryFunctionNameCH = s.shader_CH.c_str();
      }
      if (!s.shader_AH.empty()) {
        desc.hitgroup.moduleAH = module.handle;
        desc.hitgroup.entryFunctionNameAH = s.shader_AH.c_str();
      }
      if (!s.shader_IS.empty()) {
        desc.hitgroup.moduleIS = module.handle;
        desc.hitgroup.entryFunctionNameIS = s.shader_IS.c_str();
      }

      OptixProgramGroup pg;
      sizeof_log = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(optix_context, &desc, 1, &options, log, &sizeof_log, &pg));
      general_log_cb(log, sizeof_log);

      program.hitgroups.push_back(pg);
    }
  }

  program.hitgroups.shrink_to_fit();
}

/*! constructs the shader binding table */
void
OptixProgram::createSBT(OptixDeviceContext optix_context, const std::map<ObjectType, std::vector<void*>>& records)
{
  // if (records.size() != shader_hitgroups.size()) {
  //   throw std::runtime_error("expecting " + std::to_string(records.size()) + " hitgroups");
  // }

  // ------------------------------------------------------------------
  // build raygen records
  // ------------------------------------------------------------------
  std::vector<RaygenRecord> raygen_records;
  for (auto&& rg : program.raygens) {
    RaygenRecord rec = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(rg, &rec));
    rec.data = nullptr; /* for now ... */
    raygen_records.push_back(rec);
  }
  program.raygen_buffer.alloc_and_upload_async(raygen_records, /*stream=*/0);
  sbt.raygenRecord = program.raygen_buffer.d_pointer();

  // ------------------------------------------------------------------
  // build miss records
  // ------------------------------------------------------------------
  std::vector<MissRecord> miss_records;
  for (auto&& ms : program.misses) {
    MissRecord rec = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(ms, &rec));
    rec.data = nullptr; /* for now ... */
    miss_records.push_back(rec);
  }
  program.miss_buffer.alloc_and_upload_async(miss_records, /*stream=*/0);
  sbt.missRecordBase = program.miss_buffer.d_pointer();
  sbt.missRecordStrideInBytes = sizeof(MissRecord);
  sbt.missRecordCount = (int)miss_records.size();

  // ------------------------------------------------------------------
  // build hitgroup records
  // ------------------------------------------------------------------
  std::vector<HitgroupRecord> hitgroup_records;

  int shader_offset = 0;
  int sbt_offset = 0;

  // for each object type (aka. object shader group)
  for (const auto& shaders : shader_objects) {
    const auto& sbtpointers = records.at(shaders.type);

    sbt_offset_table[shaders.type] = std::vector<uint32_t>(sbtpointers.size());

    // for each object
    for (int o = 0; o < sbtpointers.size(); ++o) {

      // for each ray type
      for (int id = 0; id < shaders.hitgroup.size(); ++id) {
        HitgroupRecord rec{};
        OPTIX_CHECK(optixSbtRecordPackHeader(program.hitgroups[shader_offset + id], &rec));
        rec.data = sbtpointers[o];
        hitgroup_records.push_back(rec);
      }

      sbt_offset_table[shaders.type][o] = sbt_offset;
      sbt_offset += shaders.hitgroup.size();
    }

    shader_offset += shaders.hitgroup.size();
  }

#if 0
  std::cout << "SBT Offset Table" << std::endl;
  for (auto& t : sbt_offset_table) {
    std::cout << "  type " << object_type_string(t.first) << std::endl;
    int i = 0;
    for (auto& e : t.second) {
      std::cout << "    " << i++ << " = " << e << std::endl;
    }
  }
#endif

  program.hitgroup_buffer.alloc_and_upload_async(hitgroup_records, /*stream=*/0);
  sbt.hitgroupRecordBase = program.hitgroup_buffer.d_pointer();
  sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
  sbt.hitgroupRecordCount = (int)hitgroup_records.size();
}

void
OptixProgram::createTLAS(OptixDeviceContext optix_context,
                         const std::vector<std::vector<OptixProgram::InstanceHandler>>& blas)
{
  ias.reset(new IasData[blas.size()]);

  for (int k = 0; k < blas.size(); ++k) {
    // ==================================================================
    // Create Input
    // ==================================================================
    for (auto instance : blas[k] /* intentionally making a copy */) {
      if (sbt_offset_table.count(instance.type) > 0) {
        instance.handler.sbtOffset = sbt_offset_table[instance.type][instance.idx];
        ias[k].instances.push_back(instance.handler);
      }
      else {
        continue;
      }
    }

    if (ias[k].instances.empty())
      continue;

    ias[k].instances_buffer.alloc_and_upload_async(ias[k].instances, /*stream=*/0);

    // ==================================================================
    // Build the BVH
    // ==================================================================
    std::vector<OptixBuildInput> inputs(1);
    {
      OptixBuildInput& input = inputs[0];
      input = OptixBuildInput{};
      input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
      input.instanceArray.instances = ias[k].instances_buffer.d_pointer();
      input.instanceArray.numInstances = ias[k].instances.size();
    }

    ias[k].traversable = buildas_exec(optix_context, /*stream=*/0, inputs, ias[k].as_buffer);
  }
}

void
OptixProgram::init(OptixDeviceContext optix_context,
                   std::map<ObjectType, std::vector<void*>> records,
                   std::vector<std::vector<OptixProgram::InstanceHandler>> blas)
{
  createModule(optix_context);
  createRaygenPrograms(optix_context);
  createMissPrograms(optix_context);
  createHitgroupPrograms(optix_context);
  createPipeline(optix_context);
  createSBT(optix_context, records);
  createTLAS(optix_context, blas);
}

#endif

} // namespace ovr
