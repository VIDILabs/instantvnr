# ======================================================================== #
# Copyright 2019-2020 Qi Wu                                                #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #
# ======================================================================== #
# Copyright 2018-2019 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT target)
  set(target instantvnr)
endif()
message(STATUS "target - ${target}")

if(ENABLE_OPTIX)
  include(configure_optix)
else()
  include(configure_cuda) # skip optix here
endif()

if(DEFINED MACROCELL_SIZE_MIP)
  add_definitions(-DMACROCELL_SIZE_MIP=${MACROCELL_SIZE_MIP})
else()
  add_definitions(-DMACROCELL_SIZE_MIP=4)
endif()

if(ENABLE_FVSRN)
  add_definitions(-DKERNEL_DOUBLE_PRECISION=1) # host: double precision for unit tests
  add_definitions(-DADAPTIVE_SAMPLING=0)
  set(DISABLE_ADAPTIVE_SAMPLING ON)
endif()

# ======================================================================== #
# renderer core
# ======================================================================== #

add_library(${target} # SHARED
  renderer/method_pathtracing.cu
  renderer/method_raymarching.cu
  renderer/raytracing_shaders.h
  renderer/raytracing.h
  renderer/dda.h
  renderer/optix_program.h

  samplers/neural_sampler.cu
  samplers/neural_sampler.cpp
  samplers/neural_sampler.h

  sampler.cu
  sampler.h

  networks/tcnn_impl.cu
  network.cu
  network.h

  object.cpp
  object.h

  serializer.cpp
  serializer.h

  renderer.cpp
  renderer.h 

  types.h

  ../api.cpp
  ../api.h
)

# optix7 is an interface target anyways, so keep it public
if(ENABLE_OPTIX)
  cuda_compile_and_embed(embedded_ptx_code_optix
    renderer/method_optix.cu
  DEPENDS
    types.h
    renderer/dda.h
    renderer/raytracing.h
    renderer/raytracing_shaders.h
    renderer/method_optix.h
  LINK 
    gdt tfnmodule util
  )
  target_sources(${target} PRIVATE
    ${embedded_ptx_code_optix}
    renderer/method_optix.cu
    renderer/method_optix.cpp
  )
  target_link_libraries(${target} PUBLIC OptiX::optix7)
endif()

if(ENABLE_IN_SHADER)
  target_sources(${target} PRIVATE
    networks/tcnn_impl_decoder.cu
    networks/tcnn_impl_network.cu
  )
  target_compile_definitions(${target} PRIVATE ENABLE_IN_SHADER=1)
endif()

if(ENABLE_OUT_OF_CORE)
  target_compile_definitions(${target} PRIVATE ENABLE_OUT_OF_CORE=1)
  target_link_libraries     (${target} PRIVATE $<$<PLATFORM_ID:Linux>:aio>)
endif()

if(ENABLE_FVSRN)
  target_sources(${target} PRIVATE
    ../../fvsrn/volume_interpolation_network.cu
    networks/fvsrn_network.cu
  )
  target_compile_definitions(${target} PRIVATE ENABLE_FVSRN=1)
endif()

set_target_properties(${target} 
PROPERTIES 
  CXX_STANDARD 17
  CUDA_STANDARD 17
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/..)
target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../..)

target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=177">)   # -> variable declared but not referenced
target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=20044">) # -> extern variable '...' treated as static

target_link_libraries(${target} PRIVATE TBB::tbb)

target_link_libraries(${target} PUBLIC # intentionally keep these library public
  gdt util tfnmodule tiny-cuda-nn
)

if(TARGET openvkl::openvkl)
  target_compile_definitions(${target} PRIVATE ENABLE_OPENVKL=1)
  target_link_libraries(${target} PRIVATE openvkl::openvkl openvkl::testing)
endif()

if(DISABLE_ADAPTIVE_SAMPLING)
  target_compile_definitions(${target} PRIVATE ADAPTIVE_SAMPLING=0)
endif()

if(ENABLE_LOGGING)
  target_compile_definitions(${target} PRIVATE ENABLE_LOGGING=1)
endif()

# target_include_directories(vnr_core PUBLIC $<$<PLATFORM_ID:Windows>:${INTEL_AIO_INC}>)
# target_link_libraries     (vnr_core PUBLIC $<$<PLATFORM_ID:Windows>:${INTEL_AIO_LIB}>)
# target_compile_definitions(vnr_core PUBLIC AIO_INTEL)
# target_compile_definitions(vnr_core PRIVATE TCNN_NAMESPACE=tcnn)
