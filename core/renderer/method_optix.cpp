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

#include "method_optix.h"

namespace vnr {

extern "C" char embedded_ptx_code_optix[];

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

MethodOptiX::MethodOptiX() : OptixProgram(embedded_ptx_code_optix, RAY_TYPE_COUNT)
{
  OptixProgram::HitGroupShaders s;

  shader_raygen = "__raygen__default";

  shader_misses.push_back("__miss__radiance");
  shader_misses.push_back("__miss__shadow");

  /* objects */
  {
    OptixProgram::ObjectGroup group;
    group.type = VOLUME_STRUCTURED_REGULAR;

    s.shader_CH = "__closesthit__volume_radiance";
    s.shader_AH = "__anyhit__volume_radiance";
    s.shader_IS = "__intersection__volume";
    group.hitgroup.push_back(s);

    s.shader_CH = "__closesthit__volume_shadow";
    s.shader_AH = "__anyhit__volume_shadow";
    s.shader_IS = "__intersection__volume";
    group.hitgroup.push_back(s);

    shader_objects.push_back(group);
  }

  params_buffer.resize(sizeof(DefaultUserData), /*stream=*/0);
}

void
MethodOptiX::render(cudaStream_t stream, const LaunchParams& _params, ShadingMode mode)
{
  DefaultUserData params = _params;
  {
    params.mode = mode;
    params.traversable = ias[0].traversable;
    params.geometry_traversable = ias[1].traversable;
  }

  params_buffer.upload_async(&params, 1, stream);

  OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                          pipeline.handle,
                          stream,
                          /*! parameters and SBT */
                          params_buffer.d_pointer(),
                          params_buffer.sizeInBytes,
                          &sbt,
                          /*! dimensions of the launch: */
                          params.frame.size.x,
                          params.frame.size.y,
                          1));
}

} // namespace ovr
