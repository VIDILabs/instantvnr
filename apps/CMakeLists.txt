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

set(_instantvnr_app_common_include_dirs
  "${CMAKE_CURRENT_LIST_DIR}/../base/ovr/common"
  "${PROJECT_BINARY_DIR}/base/ovr/common"
)

function(ivnr_configure_app app_target)
  target_link_libraries(${app_target} PRIVATE rendercommon)
  target_include_directories(${app_target} PRIVATE ${_instantvnr_app_common_include_dirs})
  target_compile_options(${app_target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>
  )
  target_link_options(${app_target} PRIVATE
    "LINKER:--allow-shlib-undefined"
  )
endfunction()

# ======================================================================== #
# applications
# ======================================================================== #

if(ENABLE_OPENGL)
  add_executable(vnr_int_dual int_dual_volume.cpp)
  target_link_libraries(vnr_int_dual   PRIVATE ${target} ${GFX_LIBRARIES})
  ivnr_configure_app(vnr_int_dual)

  add_executable(vnr_int_single int_volume.cpp)
  target_link_libraries(vnr_int_single PRIVATE ${target} ${GFX_LIBRARIES})
  ivnr_configure_app(vnr_int_single)
endif()

add_executable(vnr_cmd_train  batch_trainer.cpp)
target_link_libraries(vnr_cmd_train PRIVATE ${target})
ivnr_configure_app(vnr_cmd_train)

add_executable(vnr_cmd_render batch_renderer.cpp)
target_link_libraries(vnr_cmd_render PRIVATE ${target})
ivnr_configure_app(vnr_cmd_render)

if(ENABLE_IN_SHADER)
  add_executable(vnr_cmd_isosurface batch_isosurface.cpp)
  target_link_libraries(vnr_cmd_isosurface PRIVATE ${target})
  ivnr_configure_app(vnr_cmd_isosurface)
endif()

add_executable(view_model view_model.cpp)
target_link_libraries(view_model PRIVATE ${target})
ivnr_configure_app(view_model)

if(ENABLE_IN_SHADER)
  add_executable(vnr_int_isosurface int_isosurface.cu)
  target_link_libraries(vnr_int_isosurface PUBLIC renderlib ${target} ${GFX_LIBRARIES})
  ivnr_configure_app(vnr_int_isosurface)
endif()
