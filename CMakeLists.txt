# ======================================================================== #
# Copyright 2019-2024 Qi Wu                                                #
#                                                                          #
# Licensed under the Apache License, Version 2.0                           #
# ======================================================================== #
#
# Standalone build — OVR is an in-tree submodule, TCNN is built from source
# via FetchContent, and the result is installed as a self-contained
# `instantvnr` Python package directory.
#

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif()

cmake_minimum_required(VERSION 3.24)
project(instantvnr LANGUAGES C CXX CUDA)

# OVR cmake modules (configure_build_type, configure_cxx, configure_cuda …)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/base/cmake")

include(configure_build_type)
include(configure_cxx)
include(configure_cuda)
# configure_optix is NOT included here — it unconditionally fetches OptiX headers
# even when OVR_BUILD_DEVICE_OPTIX7=OFF.  OVR handles optix internally when enabled.

###############################################################################
# GLIBCXX ABI — defaults to 1 (GCC ≥5 default); override with
#   -DIVNR_GLIBCXX_USE_CXX11_ABI=0  when linking against a library built with
#   the old ABI (e.g. PyTorch wheels that ship _GLIBCXX_USE_CXX11_ABI=0).
###############################################################################
option(IVNR_GLIBCXX_USE_CXX11_ABI "Use GLIBCXX CXX11 ABI" ON)
if(IVNR_GLIBCXX_USE_CXX11_ABI)
  set(IVNR_GLIBCXX_CXX11_ABI _GLIBCXX_USE_CXX11_ABI=1)
else()
  set(IVNR_GLIBCXX_CXX11_ABI _GLIBCXX_USE_CXX11_ABI=0)
endif()
message(STATUS "GLIBCXX_CXX11_ABI: ${IVNR_GLIBCXX_USE_CXX11_ABI}")

###############################################################################
# Staging directory — all shared libs and executables land here so the install
# step can copy them as a single package directory (same pattern as pysampler).
###############################################################################
set(_ivnr_staging "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${_ivnr_staging}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${_ivnr_staging}")
set(CMAKE_INSTALL_RPATH              "$ORIGIN")
set(CMAKE_BUILD_WITH_INSTALL_RPATH   ON)
set(CMAKE_POSITION_INDEPENDENT_CODE  ON)

###############################################################################
# OVR (OpenGL, GLFW, ImGui, GDT math, TBB, TFN module)
###############################################################################
set(OVR_BUILD_OPENGL ON CACHE BOOL "" FORCE)
set(OVR_BUILD_CUDA ON CACHE BOOL "" FORCE)
set(OVR_BUILD_DEVICE_OPTIX7 OFF CACHE BOOL "" FORCE)
set(OVR_BUILD_DEVICE_OSPRAY OFF CACHE BOOL "" FORCE)
set(OVR_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(OVR_BUILD_APPS OFF CACHE BOOL "" FORCE)
add_subdirectory(base)

###############################################################################
# TBB (may be brought in by OVR; guard against double-find)
###############################################################################
if(NOT TARGET TBB::tbb)
  find_package(TBB CONFIG REQUIRED)
endif()


###############################################################################
# tiny-cuda-nn — prefer a local checkout (tcnn/ submodule or copy), fall back
# to FetchContent when the local directory is absent.
# TCNN_REPOSITORY / TCNN_COMMIT_HASH are injected from pyproject.toml via
# scikit-build-core's cmake.define table and are used only for the fallback.
###############################################################################
set(TCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
add_subdirectory(tcnn)
# TCNN's cmake creates the `tiny-cuda-nn` target directly.
# Newer TCNN versions dropped the TCNN_NAMESPACE macro; inject it so all
# consumers get it without modifying TCNN source.
target_compile_definitions(tiny-cuda-nn PUBLIC TCNN_NAMESPACE=tcnn)

###############################################################################
# OVR compatibility aliases
###############################################################################
if((NOT TARGET util) AND (TARGET rendercommon))
  add_library(util ALIAS rendercommon)
endif()

# GFX_LIBRARIES: interactive viewer targets provided by OVR's OpenGL layer.
# OVR sets this variable; provide a safe fallback for standalone builds.
if(NOT DEFINED GFX_LIBRARIES)
  set(GFX_LIBRARIES glfwApp)
endif()

###############################################################################
# Global include paths (api.h, api_internal.h live at the project root)
###############################################################################
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

###############################################################################
# Feature flags (match references/instantvnr original defaults)
###############################################################################
set(MACROCELL_SIZE_MIP 4)
set(ENABLE_IN_SHADER   ON)
set(ENABLE_OUT_OF_CORE ON)
set(ENABLE_OPENGL      ${OVR_BUILD_OPENGL})

###############################################################################
# Subprojects
###############################################################################
add_subdirectory(core)
# device/ is the OVR plugin layer; build it as a shared library so it is
# co-installed and available for optional OVR integration.
add_subdirectory(device)
add_subdirectory(apps)
