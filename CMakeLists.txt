# ======================================================================== #
# Copyright 2019-2022 Qi Wu                                                #
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

#
# This project is designed to be compiled with OVR
#

option(OVR_BUILD_MODULE_NNVOLUME "Build Volumetric Neural Representation Device" OFF)

if(OVR_BUILD_MODULE_NNVOLUME)

  message(STATUS "Enable Neural Volume Device")

  if(DEFINED GDT_CUDA_ARCHITECTURES) 
    message(STATUS "Obtained target architecture from environment variable GDT_CUDA_ARCHITECTURES=${GDT_CUDA_ARCHITECTURES}")
    set(ENV{TCNN_CUDA_ARCHITECTURES} ${GDT_CUDA_ARCHITECTURES})
  endif()

  ################
  # tiny-cuda-nn #
  ################
  if(NOT TARGET tiny-cuda-nn)
    set(TCNN_BUILD_BENCHMARK OFF)
    set(TCNN_BUILD_EXAMPLES OFF)
    add_subdirectory(tcnn EXCLUDE_FROM_ALL)
    target_compile_definitions(tiny-cuda-nn PUBLIC    ${TCNN_DEFINITIONS})
    target_compile_definitions(tiny-cuda-nn INTERFACE TCNN_NAMESPACE=tcnn)
    target_include_directories(tiny-cuda-nn INTERFACE ${CMAKE_CURRENT_LIST_DIR}/tcnn/include)
    target_include_directories(tiny-cuda-nn INTERFACE ${CMAKE_CURRENT_LIST_DIR}/tcnn/dependencies)
    if(UNIX)
      set_target_properties(tiny-cuda-nn PROPERTIES 
        POSITION_INDEPENDENT_CODE ON
      )
    endif()
  endif()

  # Two projects uses different target names ...
  if(TARGET rendercommon)
    add_library(util ALIAS rendercommon)
  endif()

  # Simulate include paths
  include_directories(${CMAKE_CURRENT_LIST_DIR})

  ################
  #
  ################
  set(MACROCELL_SIZE_MIP 4)
  set(ENABLE_IN_SHADER ON)
  set(ENABLE_OUT_OF_CORE ON)
  set(ENABLE_OPTIX OFF)
  set(ENABLE_OPENGL ON)
  add_subdirectory(core)
  add_subdirectory(shadowmap)
  add_subdirectory(apps)

endif()
