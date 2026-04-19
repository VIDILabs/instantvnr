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
message(STATUS "target - ${target}")

include(configure_cuda) # include_guard(GLOBAL) makes this idempotent

if(DEFINED MACROCELL_SIZE_MIP)
  add_definitions(-DMACROCELL_SIZE_MIP=${MACROCELL_SIZE_MIP})
else()
  add_definitions(-DMACROCELL_SIZE_MIP=4)
endif()

if(ENABLE_FVSRN)
  add_definitions(-DKERNEL_DOUBLE_PRECISION=1) # host: double precision for unit tests
  set(DISABLE_ADAPTIVE_SAMPLING ON)
endif()

if(DISABLE_ADAPTIVE_SAMPLING)
  add_compile_definitions(ADAPTIVE_SAMPLING=0)
else()
  add_compile_definitions(ADAPTIVE_SAMPLING=1)
endif()

# ======================================================================== #
# renderer core
# ======================================================================== #

add_library(${target} SHARED
  renderer/method_pathtracing.cu
  renderer/method_pathtracing.h
  renderer/method_raymarching.cu
  renderer/method_raymarching.h
  renderer/raytracing.h
  renderer/dda.h

  samplers/neural_sampler.cu
  samplers/neural_sampler.cpp
  samplers/neural_sampler.h
  sampler.cu
  sampler.h

  network.cu
  network.h

  macrocell.cu
  macrocell.h

  instantvnr_types.cu
  instantvnr_types.h

  array.h

  serializer.cpp
  serializer.h

  renderer.cpp
  renderer.h

  ../api.cpp
  ../api.h
)

if(ENABLE_IN_SHADER)
  target_sources(${target} PRIVATE
    networks/tcnn_device_encoder.cu
    networks/tcnn_device_network.cu
    marching_cube.cu
  )
  target_compile_definitions(${target} PUBLIC ENABLE_IN_SHADER=1)
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
target_include_directories(${target} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/..>)
target_include_directories(${target} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../..>)

target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=177">)   # variable declared but not referenced
target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=20044">) # extern variable treated as static

target_link_libraries(${target} PRIVATE TBB::tbb curand tiny-cuda-nn)
target_link_libraries(${target} PUBLIC gdt util tfnmodule)

# GLIBCXX ABI compile definition — must match the installed PyTorch
target_compile_definitions(${target} PUBLIC ${IVNR_GLIBCXX_CXX11_ABI})

if(TARGET openvkl::openvkl AND TARGET openvkl::testing)
  target_compile_definitions(${target} PRIVATE ENABLE_OPENVKL=1)
  target_link_libraries(${target} PRIVATE openvkl::openvkl openvkl::testing)
endif()

if(ENABLE_LOGGING)
  target_compile_definitions(${target} PRIVATE ENABLE_LOGGING=1)
endif()
