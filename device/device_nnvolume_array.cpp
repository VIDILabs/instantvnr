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

#include "device_nnvolume_array.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>

namespace ovr::nnvolume {

namespace {

template<typename T>
std::pair<T, T>
compute_scalar_range(const void* _array, size_t count, size_t stride)
{
  static_assert(std::is_scalar<T>::value, "expecting a scalar type");

  if (stride == 0)
    stride = sizeof(T);

  T* array = (T*)_array;
  auto value = [array, stride](size_t index) -> T {
    const auto begin = (const uint8_t*)array;
    const auto curr = (T*)(begin + index * stride);
    return static_cast<T>(*curr);
  };

  T init;

  init = std::numeric_limits<T>::min();
  T actual_max = tbb::parallel_reduce(
    tbb::blocked_range<size_t>(0, count), init,
    [value](const tbb::blocked_range<size_t>& r, T v) -> T {
      for (auto i = r.begin(); i != r.end(); ++i)
        v = std::max(v, value(i));
      return v;
    },
    [](T x, T y) -> T { return std::max(x, y); });

  init = std::numeric_limits<T>::max();
  T actual_min = tbb::parallel_reduce(
    tbb::blocked_range<size_t>(0, count), init,
    [value](const tbb::blocked_range<size_t>& r, T v) -> T {
      for (auto i = r.begin(); i != r.end(); ++i)
        v = std::min(v, value(i));
      return v;
    },
    [](T x, T y) -> T { return std::min(x, y); });

  return std::make_pair(actual_min, actual_max);
}

template<typename IType, typename OType>
std::shared_ptr<char[]>
convert_array1d(const char* idata, size_t size)
{
  std::shared_ptr<char[]> odata;
  odata.reset(new char[size * sizeof(OType)]);

  tbb::parallel_for(size_t(0), size, [&](size_t idx) {
    auto* i = (IType*)&idata[idx * sizeof(IType)];
    auto* o = (OType*)&odata[idx * sizeof(OType)];
    *o = static_cast<OType>(*i);
  });

  return odata;
}

template<typename IType, typename OType>
std::shared_ptr<char[]>
convert_volume(const char* idata, vec3i dims)
{
  size_t size = dims.long_product();
  return convert_array1d<IType, OType>(idata, size);
}

template<typename InType, typename = typename std::enable_if<std::is_integral<InType>::value>::type>
std::pair<float, float>
cuda_scalar_range(const void* _array, size_t count, size_t stride)
{
  auto p = compute_scalar_range<InType>(_array, count, stride);
  return std::make_pair(integer_normalize<float, InType>(p.first), integer_normalize<float, InType>(p.second));
}

template<typename InType,
         typename = void,
         typename = typename std::enable_if<std::is_floating_point<InType>::value>::type>
std::pair<float, float>
cuda_scalar_range(const void* _array, size_t count, size_t stride)
{
  auto p = compute_scalar_range<InType>(_array, count, stride);
  return std::make_pair((float)p.first, (float)p.second);
}

} // namespace

// ------------------------------------------------------------------
// Array1DScalarCUDA
// ------------------------------------------------------------------

template<typename T>
Array1DScalarCUDA
CreateArray1DScalarCUDA(const std::vector<T>& input, cudaStream_t stream)
{
  static_assert(std::is_scalar<T>::value, "expecting a scalar type");

  Array1DScalarCUDA output;
  output.type = value_type<T>();
  output.dims = (int)input.size();
  std::tie(output.lower.v, output.upper.v) = cuda_scalar_range<T>(input.data(), input.size(), 0);
  output.scale.v = 1.f / (output.upper.v - output.lower.v);
  auto array_handler = createCudaArray1D<T>(input.data(), input.size());
  if (std::is_floating_point<T>::value) {
    output.data = createCudaTexture<T>(array_handler, cudaReadModeElementType, cudaFilterModeLinear, cudaFilterModeLinear, cudaAddressModeClamp, true);
  }
  else {
    output.data = createCudaTexture<T>(array_handler, cudaReadModeNormalizedFloat, cudaFilterModeLinear, cudaFilterModeLinear, cudaAddressModeClamp, true);
  }

  if (!output.rawptr)
    CUDA_CHECK(cudaMallocAsync((void**)&output.rawptr, output.dims.v * sizeof(T), stream));

  CUDA_CHECK(cudaMemcpyAsync(output.rawptr, (void*)input.data(), output.dims.v  * sizeof(T), cudaMemcpyHostToDevice, stream));

  return output;
}
template Array1DScalarCUDA
CreateArray1DScalarCUDA<float>(const std::vector<float>& input, cudaStream_t stream);

// This is simply a local helper function
template<typename T>
Array1DScalarCUDA
CreateArray1DScalarCUDA(array_1d_scalar_t input, const char* data)
{
  static_assert(std::is_scalar<T>::value, "expecting a scalar type");

  if (input->type != value_type<T>())
    throw std::runtime_error("type mismatch");

  Array1DScalarCUDA output;

  output.type = input->type;
  output.dims = input->dims;
  std::tie(output.lower.v, output.upper.v) = cuda_scalar_range<T>(data, input->dims.v, 0);
  output.scale.v = 1.f / (output.upper.v - output.lower.v);

  auto array_handler = createCudaArray1D<T>(data, input->dims.v);
  if (std::is_floating_point<T>::value) {
    output.data = createCudaTexture<T>(array_handler, cudaReadModeElementType, cudaFilterModeLinear, cudaFilterModeLinear, cudaAddressModeClamp, true);
  }
  else {
    output.data = createCudaTexture<T>(array_handler, cudaReadModeNormalizedFloat, cudaFilterModeLinear, cudaFilterModeLinear, cudaAddressModeClamp, true);
  }

  if (!output.rawptr)
    CUDA_CHECK(cudaMallocAsync((void**)&output.rawptr, output.dims.v * sizeof(T), 0));

  CUDA_CHECK(cudaMemcpyAsync(output.rawptr, (void*)data, output.dims.v  * sizeof(T), cudaMemcpyHostToDevice, 0));

  return output;
}

Array1DScalarCUDA
CreateArray1DScalarCUDA(array_1d_scalar_t input)
{
  Array1DScalarCUDA output;
  std::shared_ptr<char[]> buffer;

  switch (input->type) {
  case VALUE_TYPE_UINT8: output = CreateArray1DScalarCUDA<uint8_t>(input, input->data()); break;
  case VALUE_TYPE_INT8: output = CreateArray1DScalarCUDA<int8_t>(input, input->data()); break;
  case VALUE_TYPE_UINT32: output = CreateArray1DScalarCUDA<uint32_t>(input, input->data()); break;
  case VALUE_TYPE_INT32: output = CreateArray1DScalarCUDA<int32_t>(input, input->data()); break;
  case VALUE_TYPE_FLOAT: output = CreateArray1DScalarCUDA<float>(input, input->data()); break;

  case VALUE_TYPE_UINT16:
    buffer = convert_array1d<uint16_t, float>(input->data(), input->dims.v);
    input->type = VALUE_TYPE_FLOAT;
    output = CreateArray1DScalarCUDA<float>(input, input->data());
    break;
  case VALUE_TYPE_INT16:
    buffer = convert_array1d<int16_t, float>(input->data(), input->dims.v);
    input->type = VALUE_TYPE_FLOAT;
    output = CreateArray1DScalarCUDA<float>(input, input->data());
    break;
  case VALUE_TYPE_DOUBLE:
    buffer = convert_array1d<double, float>(input->data(), input->dims.v);
    input->type = VALUE_TYPE_FLOAT;
    output = CreateArray1DScalarCUDA<float>(input, input->data());
    break;

  default: throw std::runtime_error("[nnvolume] unexpected array type ...");
  }

  return output;
}

// ------------------------------------------------------------------
// Array1DFloat4CUDA
// ------------------------------------------------------------------

Array1DFloat4CUDA
CreateArray1DFloat4CUDA(const std::vector<vec4f>& input, cudaStream_t stream)
{
  Array1DFloat4CUDA output;

  output.type = VALUE_TYPE_FLOAT4;
  output.dims = (int)input.size();
  auto* dx = (float*)input.data();
  auto* dy = dx + 1;
  auto* dz = dx + 2;
  auto* dw = dx + 3;
  std::tie(output.lower.x, output.upper.x) = cuda_scalar_range<float>(dx, input.size(), sizeof(float4));
  std::tie(output.lower.y, output.upper.y) = cuda_scalar_range<float>(dy, input.size(), sizeof(float4));
  std::tie(output.lower.z, output.upper.z) = cuda_scalar_range<float>(dz, input.size(), sizeof(float4));
  std::tie(output.lower.w, output.upper.w) = cuda_scalar_range<float>(dw, input.size(), sizeof(float4));
  auto array_handler = createCudaArray1D<float4>(input.data(), input.size());
  output.data = createCudaTexture<float4>(array_handler, cudaReadModeElementType, cudaFilterModeLinear, cudaFilterModeLinear, cudaAddressModeClamp, true);
  output.scale.x = 1.f / (output.upper.x - output.lower.x);
  output.scale.y = 1.f / (output.upper.y - output.lower.y);
  output.scale.z = 1.f / (output.upper.z - output.lower.z);
  output.scale.w = 1.f / (output.upper.w - output.lower.w);

  if (!output.rawptr)
    CUDA_CHECK(cudaMallocAsync((void**)&output.rawptr, output.dims.long_product() * sizeof(vec4f), stream));

  CUDA_CHECK(cudaMemcpyAsync(output.rawptr, (void*)input.data(), output.dims.long_product()  * sizeof(vec4f), cudaMemcpyHostToDevice, stream));

  return output;
}

Array1DFloat4CUDA
CreateArray1DFloat4CUDA(array_1d_float4_t input)
{
  if (input->type != value_type<vec4f>())
    throw std::runtime_error("type mismatch");

  Array1DFloat4CUDA output;

  auto* dx = (float*)input->data();
  auto* dy = dx + 1;
  auto* dz = dx + 2;
  auto* dw = dx + 3;
  std::tie(output.lower.x, output.upper.x) = cuda_scalar_range<float>(dx, input->dims.v, sizeof(float4));
  std::tie(output.lower.y, output.upper.y) = cuda_scalar_range<float>(dy, input->dims.v, sizeof(float4));
  std::tie(output.lower.z, output.upper.z) = cuda_scalar_range<float>(dz, input->dims.v, sizeof(float4));
  std::tie(output.lower.w, output.upper.w) = cuda_scalar_range<float>(dw, input->dims.v, sizeof(float4));

  output.type = input->type;
  output.dims = input->dims;

  auto array_handler = createCudaArray1D<float4>(input->data(), input->dims.v);
  output.data = createCudaTexture<float4>(array_handler, cudaReadModeElementType, cudaFilterModeLinear, cudaFilterModeLinear, cudaAddressModeClamp, true);

  output.scale.x = 1.f / (output.upper.x - output.lower.x);
  output.scale.y = 1.f / (output.upper.y - output.lower.y);
  output.scale.z = 1.f / (output.upper.z - output.lower.z);
  output.scale.w = 1.f / (output.upper.w - output.lower.w);

  if (!output.rawptr)
    CUDA_CHECK(cudaMallocAsync((void**)&output.rawptr, output.dims.long_product() * sizeof(vec4f), 0));

  CUDA_CHECK(cudaMemcpyAsync(output.rawptr, (void*)input->data(), output.dims.long_product()  * sizeof(vec4f), cudaMemcpyHostToDevice, 0));

  return output;
}

// ------------------------------------------------------------------
// Array3DScalarCUDA
// ------------------------------------------------------------------

template<typename T>
Array3DScalarCUDA
CreateArray3DScalarCUDA(void* input, vec3i dims)
{
  const size_t elem_count = (size_t)dims.x * dims.y * dims.z;

  Array3DScalarCUDA output;

  output.type = value_type<T>();
  output.dims = dims;
  std::tie(output.lower.v, output.upper.v) = cuda_scalar_range<T>(input, elem_count, 0);
  output.scale.v = 1.f / (output.upper.v - output.lower.v);

  auto array_handler = createCudaArray3D<T>(input, (int3&)dims);
  if (std::is_floating_point<T>::value) {
    output.data = createCudaTexture<T>(array_handler, cudaReadModeElementType, cudaFilterModeLinear, cudaFilterModeLinear, cudaAddressModeClamp, true);
  }
  else {
    output.data = createCudaTexture<T>(array_handler, cudaReadModeNormalizedFloat, cudaFilterModeLinear, cudaFilterModeLinear, cudaAddressModeClamp, true);
  }

  return output;
}

#define instantiate_create_array_3d_scalar(T) \
  template Array3DScalarCUDA CreateArray3DScalarCUDA<T>(void* input, vec3i dims);

instantiate_create_array_3d_scalar(int8_t);
instantiate_create_array_3d_scalar(uint8_t);
instantiate_create_array_3d_scalar(uint32_t);
instantiate_create_array_3d_scalar(int32_t);
instantiate_create_array_3d_scalar(float);

#undef instantiate_create_array_3d_scalar

Array3DScalarCUDA
CreateArray3DScalarCUDA(array_3d_scalar_t array)
{
  Array3DScalarCUDA output;
  std::shared_ptr<char[]> buffer;

  switch (array->type) {
  case VALUE_TYPE_UINT8: output = CreateArray3DScalarCUDA<uint8_t>(array->data(), array->dims); break;
  case VALUE_TYPE_INT8: output = CreateArray3DScalarCUDA<int8_t>(array->data(), array->dims); break;
  case VALUE_TYPE_UINT32: output = CreateArray3DScalarCUDA<uint32_t>(array->data(), array->dims); break;
  case VALUE_TYPE_INT32: output = CreateArray3DScalarCUDA<int32_t>(array->data(), array->dims); break;
  case VALUE_TYPE_FLOAT: output = CreateArray3DScalarCUDA<float>(array->data(), array->dims); break;
  // TODO cannot handle the following correctly, so converting them into floats //
  case VALUE_TYPE_UINT16:
    buffer = convert_volume<uint16_t, float>(array->data(), array->dims);
    output = CreateArray3DScalarCUDA<float>(buffer.get(), array->dims);
    break;
  case VALUE_TYPE_INT16:
    buffer = convert_volume<int16_t, float>(array->data(), array->dims);
    output = CreateArray3DScalarCUDA<float>(buffer.get(), array->dims);
    break;
  case VALUE_TYPE_DOUBLE:
    buffer = convert_volume<double, float>(array->data(), array->dims);
    output = CreateArray3DScalarCUDA<float>(buffer.get(), array->dims);
    break;
  default: throw std::runtime_error("[nnvolume] unexpected volume type ...");
  }

  return output;
}

} // namespace ovr::nnvolume
