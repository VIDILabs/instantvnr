//. ======================================================================== //
//. Copyright 2019-2020 Qi Wu                                                //
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

#pragma once

#include "ovr/scene.h"

#include <cuda_runtime.h>
#include <cuda/cuda_buffer.h>

#if defined(__cplusplus)
#include <type_traits>
#endif // defined(__cplusplus)

namespace ovr {

ovr_instantiate_value_type_function(VALUE_TYPE_FLOAT2, float2);
ovr_instantiate_value_type_function(VALUE_TYPE_FLOAT3, float3);
ovr_instantiate_value_type_function(VALUE_TYPE_FLOAT4, float4);

namespace nnvolume {

template<int DIM, int DIM_ELEMENT>
struct ArrayCUDA {
  ValueType type;
  math::vec_t<int, DIM> dims{ 0 };
  /* normalized value ranges */
  math::vec_t<float, DIM_ELEMENT> lower{ 0 }; // value range for the texture
  math::vec_t<float, DIM_ELEMENT> upper{ 0 }; // value range for the texture
  math::vec_t<float, DIM_ELEMENT> scale{ 1 };
  cudaTextureObject_t data{}; // the storage of the data on texture unit
  void* rawptr{ nullptr };
};

using Array1DScalarCUDA = ArrayCUDA<1, 1>;
using Array1DFloat4CUDA = ArrayCUDA<1, 4>;
using Array3DScalarCUDA = ArrayCUDA<3, 1>;

struct DeviceTransferFunction {
  Array1DFloat4CUDA color;
  Array1DScalarCUDA opacity;
  vec2f value_range;
  float range_rcp_norm; // == 1.f / value_range.span()
};

#if defined(__cplusplus)
// clang-format off

// integer normalization following OpenGL's specification
// https://www.khronos.org/opengl/wiki/Normalized_Integer

// normalize unsigned integer
template<typename OutType, typename InType,
         typename = typename std::enable_if<std::is_floating_point<OutType>::value && 
                                            std::is_integral<InType>::value &&
                                            std::is_unsigned<InType>::value>::type>
inline OutType
integer_normalize(InType val)
{
  const auto maxVal = static_cast<OutType>(std::numeric_limits<InType>::max());
  return (static_cast<OutType>(val) / maxVal);
}

// normalize signed integer
template<typename OutType, typename InType, typename = void,
         typename = typename std::enable_if<std::is_floating_point<OutType>::value && 
                                            std::is_integral<InType>::value &&
                                            std::is_signed<InType>::value>::type>
inline OutType
integer_normalize(InType val)
{
  const auto maxVal = static_cast<OutType>(std::numeric_limits<InType>::max());
  const OutType normVal = static_cast<OutType>(val) / maxVal;
  return (normVal < OutType(-1.0) ? OutType(-1.0) : normVal);
}

inline float
integer_normalize(float value, ValueType type)
{
  switch (type) {
  case VALUE_TYPE_UINT8:  return integer_normalize<float, uint8_t >((uint8_t)value);
  case VALUE_TYPE_INT8:   return integer_normalize<float, int8_t  >((int8_t)value);
  case VALUE_TYPE_UINT16: return integer_normalize<float, uint16_t>((uint16_t)value);
  case VALUE_TYPE_INT16:  return integer_normalize<float, int16_t >((int16_t)value);
  case VALUE_TYPE_UINT32: return integer_normalize<float, uint32_t>((uint32_t)value);
  case VALUE_TYPE_INT32:  return integer_normalize<float, int32_t >((int32_t)value);
  case VALUE_TYPE_FLOAT:  return value;
  case VALUE_TYPE_DOUBLE: return (float)value;
  default: throw std::runtime_error("unknown type conversion");
  }
}  

// clang-format on

// factory functions //

template<typename T>
Array1DScalarCUDA
CreateArray1DScalarCUDA(const std::vector<T>& input, cudaStream_t stream = 0);
Array1DScalarCUDA
CreateArray1DScalarCUDA(array_1d_scalar_t input);

Array1DFloat4CUDA
CreateArray1DFloat4CUDA(const std::vector<vec4f>& input, cudaStream_t stream = 0);
Array1DFloat4CUDA
CreateArray1DFloat4CUDA(array_1d_float4_t input);

template<typename T>
Array3DScalarCUDA
CreateArray3DScalarCUDA(void* input, vec3i dims);
Array3DScalarCUDA
CreateArray3DScalarCUDA(array_3d_scalar_t input);

template<typename Type>
inline cudaArray_t
createCudaArray1D(const void* dataPtr, const size_t& size)
{
  cudaArray_t dataArr{};

  // Allocate actually a 2D CUDA array of shape N x 1
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<Type>();
  CUDA_CHECK(cudaMallocArray(&dataArr, &channel_desc, size, 1));

  // Copy data to the CUDA array
  const size_t nByte = size * sizeof(Type);
  CUDA_CHECK(cudaMemcpy2DToArray(dataArr, 0, 0, dataPtr, nByte, nByte, 1, cudaMemcpyHostToDevice));
  return dataArr;

  // Need to bind the CUDA array to the texture object
}

#endif // defined(__cplusplus)

} // namespace nnvolume
} // namespace ovr
