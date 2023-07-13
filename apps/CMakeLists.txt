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

# ======================================================================== #
# applications
# ======================================================================== #
if(ENABLE_OPENGL)
  add_executable(vnr_int_dual int_dual_volume.cpp)
  target_link_libraries(vnr_int_dual   PRIVATE ${target} ${GFX_LIBRARIES})

  add_executable(vnr_int_single int_volume.cpp)
  target_link_libraries(vnr_int_single PRIVATE ${target} ${GFX_LIBRARIES})
endif()

add_executable(vnr_cmd_train  batch_trainer.cpp)
target_link_libraries(vnr_cmd_train PRIVATE ${target})

add_executable(vnr_cmd_render batch_renderer.cpp)
target_link_libraries(vnr_cmd_render PRIVATE ${target})

add_executable(vnr_cmd_isosurface batch_isosurface.cpp)
target_link_libraries(vnr_cmd_isosurface PRIVATE ${target})

add_executable(view_model view_model.cpp)
target_link_libraries(view_model PRIVATE ${target})

# OVR application
if(DEFINED OVR_BUILD_MODULE_NNVOLUME)
  add_executable(generate_shadow_map shadowmap.cu)
  target_link_libraries(generate_shadow_map 
    PUBLIC 
      renderlib glfwApp
      $<TARGET_OBJECTS:device_nnvolume_array>
  )
  add_executable(vnr_int_isosurface int_isosurface.cu)
  target_link_libraries(vnr_int_isosurface 
  PUBLIC 
    renderlib # TBB::tbb 
    ${target} ${GFX_LIBRARIES} 
  )
endif()
