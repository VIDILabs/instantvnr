//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //

/**
 * Geometry Types Defined by the Application
 */
#pragma once

#include <cuda_runtime.h>

#if defined(ENABLE_OPTIX)
#include <optix.h>
#include <optix_stubs.h>
#endif

#include "mathdef.h"

#include <array>
#include <vector>
#include <cstdio>

// ------------------------------------------------------------------
// Array Definitions
// ------------------------------------------------------------------

namespace vnr {

struct Array1D
{
  ValueType type;
  size_t length{ 0 };
  cudaTextureObject_t data{ 0 }; // the storage of the data on texture unit
  void* rawptr{ nullptr };
};

struct Array3DScalar {
  ValueType type;
  vec3i dims { 0 };
  vec3f rdims{ 0 };
  cudaTextureObject_t data{}; // the storage of the data on texture unit
};

using Array1DScalar = Array1D;
using Array1DFloat4 = Array1D;

// ------------------------------------------------------------------
// Helper Functions
// ------------------------------------------------------------------
template<typename Type>
static void // copy from CUDA linear memory to CUDA array
CopyLinearMemoryToArray(void* src_pointer, cudaArray_t dst_array, vec3i dims, cudaMemcpyKind kind)
{
  cudaMemcpy3DParms param = { 0 };
  param.srcPos = make_cudaPos(0, 0, 0);
  param.dstPos = make_cudaPos(0, 0, 0);
  param.srcPtr = make_cudaPitchedPtr(src_pointer, dims.x * sizeof(Type), dims.x, dims.y);
  param.dstArray = dst_array;
  param.extent = make_cudaExtent(dims.x, dims.y, dims.z);
  param.kind = kind;
  CUDA_CHECK(cudaMemcpy3D(&param));
}

template<typename T>
static void
CreateArray3DScalar(cudaArray_t& array, cudaTextureObject_t& tex, vec3i dims, bool trilinear, void* ptr = nullptr)
{
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<T>();
  CUDA_CHECK(cudaMalloc3DArray(&array, &channel_desc, make_cudaExtent(dims.x, dims.y, dims.z)));
  util::total_n_bytes_allocated() += (size_t)dims.x * dims.y * dims.z * sizeof(T);
#ifdef VNR_VERBOSE_MEMORY_ALLOCS
	std::cout << "3D Texture: Allocating " << util::prettyBytes((size_t)dims.x * dims.y * dims.z * sizeof(T)) << std::endl;
#endif
  const auto filter = trilinear ? cudaFilterModeLinear : cudaFilterModePoint;
  tex = createCudaTexture<T>(array, cudaReadModeElementType, filter, filter);
  if (ptr) CopyLinearMemoryToArray<T>((T*)ptr, array, dims, cudaMemcpyHostToDevice);
}

template<typename T> 
static void
CreateArray1DScalar(cudaStream_t stream, const std::vector<T>& input, cudaArray_t& array_handler, Array1DScalar& output)
{
  static_assert(std::is_scalar<T>::value, "expecting a scalar type");

  output.type = value_type<T>();
  output.length = input.size();

  cudaChannelFormatDesc desc; cudaExtent extent{}; unsigned int flags;
  if (array_handler) {
    CUDA_CHECK(cudaArrayGetInfo(&desc, &extent, &flags, array_handler));
  }

  if (extent.width != input.size()) {
    if (array_handler) {
      CUDA_CHECK(cudaFreeArray(array_handler));
      array_handler = 0;
    }
    if (output.data) {
      CUDA_CHECK(cudaDestroyTextureObject(output.data));
      output.data = 0;
    }
    if (output.rawptr) {
      CUDA_CHECK(cudaFree(output.rawptr));
      output.rawptr = 0;
    }
  }

  if (!array_handler) {
    array_handler = allocateCudaArray1D<T>(input.data(), input.size());
  }
  else {
    fillCudaArray1D<T>(array_handler, input.data(), input.size());
  }

  if (!output.data) {
    output.data = createCudaTexture<T>(array_handler, cudaReadModeElementType, cudaFilterModeLinear);
  }

  if (!output.rawptr) {
    CUDA_CHECK(cudaMallocAsync((void**)&output.rawptr, output.length * sizeof(T), stream));
    util::total_n_bytes_allocated() += output.length * sizeof(T);
#ifdef VNR_VERBOSE_MEMORY_ALLOCS
    std::cout << "[mem] Linear " << util::prettyBytes(output.length * sizeof(T)) << std::endl;
#endif
  }
  CUDA_CHECK(cudaMemcpyAsync(output.rawptr, (void*)input.data(), output.length  * sizeof(T), cudaMemcpyHostToDevice, stream));
}

static void
CreateArray1DFloat4(cudaStream_t stream, const std::vector<float4>& input, cudaArray_t& array_handler, Array1DFloat4& output)
{
  output.type = VALUE_TYPE_FLOAT4;
  output.length = input.size();

  cudaChannelFormatDesc desc; cudaExtent extent{}; unsigned int flags;
  if (array_handler) {
    CUDA_CHECK(cudaArrayGetInfo(&desc, &extent, &flags, array_handler));
  }

  if (extent.width != input.size()) {
    if (array_handler) {
      CUDA_CHECK(cudaFreeArray(array_handler));
      array_handler = 0;
    }
    if (output.data) {
      CUDA_CHECK(cudaDestroyTextureObject(output.data));
      output.data = 0;
    }
    if (output.rawptr) {
      CUDA_CHECK(cudaFree(output.rawptr));
      output.rawptr = 0;
    }
  }

  if (!array_handler) {
    array_handler = allocateCudaArray1D<float4>(input.data(), input.size());
  }
  else {
    fillCudaArray1D<float4>(array_handler, input.data(), input.size());
  }

  if (!output.data) {
    output.data = createCudaTexture<float4>(array_handler, cudaReadModeElementType, cudaFilterModeLinear);
  }

  if (!output.rawptr) {
    CUDA_CHECK(cudaMallocAsync((void**)&output.rawptr, output.length * sizeof(float4), stream));
    util::total_n_bytes_allocated() += output.length * sizeof(float4);
#ifdef VNR_VERBOSE_MEMORY_ALLOCS
	  std::cout << "[mem] Linear " << util::prettyBytes(output.length * sizeof(float4)) << std::endl;
#endif
  }
  CUDA_CHECK(cudaMemcpyAsync(output.rawptr, (void*)input.data(), output.length  * sizeof(float4), cudaMemcpyHostToDevice, stream));
  // CUDA_SYNC_CHECK();
}

static void
CreateArray1DFloat4(cudaStream_t stream, const std::vector<vec4f>& input, cudaArray_t& array_handler, Array1DFloat4& output)
{
  return CreateArray1DFloat4(stream, (const std::vector<float4>&)input, array_handler, output);
}

} // namespace vnr
