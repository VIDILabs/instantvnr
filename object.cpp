//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //

#include "object.h"

namespace vidi {
enum VoxelType {
  VOXEL_UINT8  = vnr::VALUE_TYPE_UINT8,
  VOXEL_INT8   = vnr::VALUE_TYPE_INT8,
  VOXEL_UINT16 = vnr::VALUE_TYPE_UINT16,
  VOXEL_INT16  = vnr::VALUE_TYPE_INT16,
  VOXEL_UINT32 = vnr::VALUE_TYPE_UINT32,
  VOXEL_INT32  = vnr::VALUE_TYPE_INT32,
  VOXEL_FLOAT  = vnr::VALUE_TYPE_FLOAT,
  VOXEL_DOUBLE = vnr::VALUE_TYPE_DOUBLE,
};
}
#define VIDI_VOLUME_EXTERNAL_TYPE_ENUM
#include <vidi_volume_reader.h>

#include <vidi_parallel_algorithm.h>

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <fstream>
#include <memory>

using vidi::reverse_byte_order;

namespace vnr {

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------
namespace {

template<typename IType, typename OType>
std::shared_ptr<char[]>
convert_volume(std::shared_ptr<char[]> idata, size_t size)
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

} // namespace

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------
#if defined(ENABLE_OPTIX)
OptixTraversableHandle
buildas_exec(OptixDeviceContext optixContext,
             cudaStream_t stream,
             std::vector<OptixBuildInput> input,
             CUDABuffer& asBuffer)
{
  OptixTraversableHandle asHandle{ 0 };

  //
  // BLAS setup
  //

  OptixAccelBuildOptions accelOptions = {};
  accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes blasBufferSizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext,
                                           &accelOptions,
                                           input.data(),
                                           (int)input.size(), // num_build_inputs
                                           &blasBufferSizes));

  //
  // prepare compaction
  //

  CUDABuffer compactedSizeBuffer;
  compactedSizeBuffer.alloc(sizeof(uint64_t), stream);

  OptixAccelEmitDesc emitDesc;
  emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emitDesc.result = compactedSizeBuffer.d_pointer();

  //
  // execute build (main stage)
  //

  CUDABuffer tempBuffer;
  tempBuffer.alloc(blasBufferSizes.tempSizeInBytes, stream);

  CUDABuffer outputBuffer;
  outputBuffer.alloc(blasBufferSizes.outputSizeInBytes, stream);

  OPTIX_CHECK(optixAccelBuild(optixContext,
                              stream,
                              &accelOptions,
                              input.data(),
                              (int)input.size(),
                              tempBuffer.d_pointer(),
                              tempBuffer.sizeInBytes,
                              outputBuffer.d_pointer(),
                              outputBuffer.sizeInBytes,
                              &asHandle,
                              &emitDesc,
                              1));
  CUDA_SYNC_CHECK();

  //
  // perform compaction
  //

  uint64_t compactedSize;
  compactedSizeBuffer.download_async(&compactedSize, 1, stream);

  asBuffer.alloc(compactedSize, stream);
  OPTIX_CHECK(optixAccelCompact(optixContext, stream, asHandle, asBuffer.d_pointer(), asBuffer.sizeInBytes, &asHandle));
  CUDA_SYNC_CHECK();

  //
  // aaaaaand .... clean up
  //

  outputBuffer.free(stream); // << the UNcompacted, temporary output buffer
  tempBuffer.free(stream);
  compactedSizeBuffer.free(stream);

  return asHandle;
}
#endif

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

/*! compute 3x4 transformation matrix */
void
InstantiableGeometry::transform(float transform[12]) const
{
  transform[0] = matrix.l.row0().x;
  transform[1] = matrix.l.row0().y;
  transform[2] = matrix.l.row0().z;
  transform[3] = matrix.p.x;
  transform[4] = matrix.l.row1().x;
  transform[5] = matrix.l.row1().y;
  transform[6] = matrix.l.row1().z;
  transform[7] = matrix.p.y;
  transform[8] = matrix.l.row2().x;
  transform[9] = matrix.l.row2().y;
  transform[10] = matrix.l.row2().z;
  transform[11] = matrix.p.z;
}

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------
#if defined(ENABLE_OPTIX)

OptixTraversableHandle
AabbGeometry::buildas(OptixDeviceContext optixContext, cudaStream_t stream)
{
  aabbBuffer.alloc_and_upload_async(&aabb, 1, stream);

  CUdeviceptr d_aabb = aabbBuffer.d_pointer();
  uint32_t f_aabb = 0;

  OptixBuildInput volumeInput = {}; // use one AABB input
  volumeInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
#if OPTIX_ABI_VERSION < 23
  auto& customPrimitiveArray = volumeInput.aabbArray;
#else
  auto& customPrimitiveArray = volumeInput.customPrimitiveArray;
#endif
  customPrimitiveArray.aabbBuffers = &d_aabb;
  customPrimitiveArray.numPrimitives = 1;
  customPrimitiveArray.strideInBytes = 0;
  customPrimitiveArray.primitiveIndexOffset = 0;
  customPrimitiveArray.flags = &f_aabb;
  customPrimitiveArray.numSbtRecords = 1;
  customPrimitiveArray.sbtIndexOffsetBuffer = 0;
  customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
  customPrimitiveArray.sbtIndexOffsetStrideInBytes = 0;

  std::vector<OptixBuildInput> inputs = { volumeInput };
  return buildas_exec(optixContext, stream, inputs, asBuffer);
}

OptixTraversableHandle
MeshGeometry::buildas(OptixDeviceContext optixContext, cudaStream_t stream)
{
  // upload the model to the device: the builder
  vertexBuffer.alloc_and_upload_async(vertex, stream);
  indexBuffer.alloc_and_upload_async(index, stream);

  OptixBuildInput triangleInput{};
  triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

  // // create local variables, because we need a *pointer* to the device pointers
  // auto* d_vertex = (vec3f*)vertexBuffer.d_pointer();
  // auto* d_index = (vec3i*)indexBuffer.d_pointer();

  triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  triangleInput.triangleArray.vertexStrideInBytes = sizeof(vec3f);
  triangleInput.triangleArray.numVertices = (int)vertex.size();
  triangleInput.triangleArray.vertexBuffers = (const CUdeviceptr*)&vertexBuffer.d_pointer();

  triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  triangleInput.triangleArray.indexStrideInBytes = sizeof(vec3i);
  triangleInput.triangleArray.numIndexTriplets = (int)index.size();
  triangleInput.triangleArray.indexBuffer = (CUdeviceptr)indexBuffer.d_pointer();

  asBuildflag = 0;

  // in this example we have one SBT entry, and no per-primitive materials:
  triangleInput.triangleArray.flags = &asBuildflag;
  triangleInput.triangleArray.numSbtRecords = 1;
  triangleInput.triangleArray.sbtIndexOffsetBuffer = 0;
  triangleInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
  triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

  std::vector<OptixBuildInput> inputs = { triangleInput };
  return buildas_exec(optixContext, stream, inputs, asBuffer);
}

#endif

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

StructuredRegularVolume::~StructuredRegularVolume()
{
  if (tfn_color_array_handler) {
    CUDA_CHECK_NOEXCEPT(cudaFreeArray(tfn_color_array_handler));
    tfn_color_array_handler = NULL;
    util::total_n_bytes_allocated() -= self.tfn.colors.length * sizeof(float4);
#ifdef VNR_VERBOSE_MEMORY_ALLOCS
    printf("[mem] Array1D free %s\n", util::prettyBytes(self.tfn.colors.length * sizeof(float4)).c_str());
#endif
  }
  if (self.tfn.colors.data) {
    CUDA_CHECK_NOEXCEPT(cudaDestroyTextureObject(self.tfn.colors.data));
    self.tfn.colors.data = { 0 };
  }
  if (self.tfn.colors.rawptr) {
    CUDA_CHECK_NOEXCEPT(cudaFree(self.tfn.colors.rawptr));
    self.tfn.colors.rawptr = nullptr;
    util::total_n_bytes_allocated() -= self.tfn.colors.length * sizeof(float4);
#ifdef VNR_VERBOSE_MEMORY_ALLOCS
    printf("[mem] Linear free %s\n", util::prettyBytes(self.tfn.colors.length * sizeof(float4)).c_str());
#endif
  }
  self.tfn.colors.length = 0;

  if (tfn_alpha_array_handler) {
    CUDA_CHECK_NOEXCEPT(cudaFreeArray(tfn_alpha_array_handler));
    tfn_color_array_handler = NULL;
    util::total_n_bytes_allocated() -= self.tfn.alphas.length * sizeof(float);
#ifdef VNR_VERBOSE_MEMORY_ALLOCS
    printf("[mem] Array1D free %s\n", util::prettyBytes(self.tfn.alphas.length * sizeof(float)).c_str());
#endif
  }
  if (self.tfn.alphas.data) {
    CUDA_CHECK_NOEXCEPT(cudaDestroyTextureObject(self.tfn.alphas.data));
    self.tfn.alphas.data = { 0 };
  }
  if (self.tfn.alphas.rawptr) {
    CUDA_CHECK_NOEXCEPT(cudaFree(self.tfn.alphas.rawptr));
    self.tfn.alphas.rawptr = nullptr;
    util::total_n_bytes_allocated() -= self.tfn.alphas.length * sizeof(float);
#ifdef VNR_VERBOSE_MEMORY_ALLOCS
    printf("[mem] Linear free %s\n", util::prettyBytes(self.tfn.alphas.length * sizeof(float)).c_str());
#endif
  }
  self.tfn.alphas.length = 0;
}

CUdeviceptr
StructuredRegularVolume::get_sbt_pointer(cudaStream_t stream)
{
  if (!GetSbtPtr())
    return CreateSbtPtr(stream); /* upload to GPU */
  return GetSbtPtr();
}

void
StructuredRegularVolume::commit(cudaStream_t stream)
{
  self.step = 1.f / sampling_rate;
  self.step_rcp = sampling_rate;
  self.grad_step = vec3f(1.f / vec3f(self.volume.dims));

  UpdateSbtData(stream);
}

void 
StructuredRegularVolume::set_macrocell(vec3i dims, vec3f spacings, vec2f* d_value_range, float* d_max_opacity)
{
  self.macrocell_value_range = d_value_range;
  self.macrocell_max_opacity = d_max_opacity;

  self.macrocell_dims = dims;
  self.macrocell_spacings = spacings;
  self.macrocell_spacings_rcp = 1.f / spacings;
}

void
StructuredRegularVolume::set_transfer_function(cudaStream_t stream, const std::vector<vec3f>& c, const std::vector<vec2f>& o, const range1f& r)
{
  colors_data.resize(c.size());
  for (int i = 0; i < colors_data.size(); ++i) {
    colors_data[i].x = c[i].x;
    colors_data[i].y = c[i].y;
    colors_data[i].z = c[i].z;
    colors_data[i].w = 1.f;
  }
  alphas_data.resize(o.size());
  for (int i = 0; i < alphas_data.size(); ++i) {
    alphas_data[i] = o[i].y;
  }

  if (!colors_data.empty())
    CreateArray1DFloat4(stream, colors_data, tfn_color_array_handler, self.tfn.colors);
  if (!alphas_data.empty())
    CreateArray1DScalar(stream, alphas_data, tfn_alpha_array_handler, self.tfn.alphas);

  // set_value_range(r.x, r.y);

  if (!r.is_empty()) {
    self.tfn.range.upper = min(original_data_range.upper, r.upper);
    self.tfn.range.lower = max(original_data_range.lower, r.lower);
  }
  self.tfn.range_rcp_norm = 1.f / self.tfn.range.span();
}

void
StructuredRegularVolume::set_sampling_rate(float r)
{
  sampling_rate = r;
}

void StructuredRegularVolume::set_density_scale(float scale)
{
  self.density_scale = scale;
}

void
StructuredRegularVolume::set_volume(Array3DScalar& v)
{
  self.volume = v;
}

void 
StructuredRegularVolume::set_volume(cudaTextureObject_t data)
{
  Array3DScalar& output = self.volume;
  output.data = data;
}

void
StructuredRegularVolume::set_volume(cudaTextureObject_t data, ValueType type, vec3i dims, range1f range)
{
  Array3DScalar& output = self.volume;
  output.dims = dims;
  output.data = data;
  output.type = type;

  original_data_range = range; /* should be [0,1] */
}

void 
StructuredRegularVolume::set_clipping(vec3f lower, vec3f upper)
{
  self.bbox.lower = lower;
  self.bbox.upper = upper;
}

} // namespace ovr
