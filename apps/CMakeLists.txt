# ======================================================================== #
# Copyright 2019-2022 Qi Wu                                                #
# Licensed under the Apache License, Version 2.0                           #
# ======================================================================== #
# ======================================================================== #
# Copyright 2018-2019 Ingo Wald                                            #
# Licensed under the Apache License, Version 2.0                           #
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

if(ENABLE_IN_SHADER)
  add_executable(vnr_cmd_isosurface batch_isosurface.cpp)
  target_link_libraries(vnr_cmd_isosurface PRIVATE ${target})
endif()

add_executable(view_model view_model.cpp)
target_link_libraries(view_model PRIVATE ${target})

# OVR plugin applications — only built when instantvnr is loaded as an OVR device
if(DEFINED OVR_BUILD_MODULE_NNVOLUME)

  add_executable(generate_shadow_map shadowmap.cu)
  target_link_libraries(generate_shadow_map PUBLIC renderlib glfwApp $<TARGET_OBJECTS:device_nnvolume_array>)

  if(ENABLE_IN_SHADER)
    add_executable(vnr_int_isosurface int_isosurface.cu)
    target_link_libraries(vnr_int_isosurface PUBLIC renderlib ${target} ${GFX_LIBRARIES})
  endif()

endif()
