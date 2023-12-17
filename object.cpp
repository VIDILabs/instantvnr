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

StructuredRegularVolume::~StructuredRegularVolume()
{
  if (tfn_color_array_handler) {
    CUDA_CHECK_NOEXCEPT(cudaTrackedFreeArray(tfn_color_array_handler));
    tfn_color_array_handler = NULL;
//     util::total_n_bytes_allocated() -= self.tfn.colors.length * sizeof(float4);
// #ifdef VNR_VERBOSE_MEMORY_ALLOCS
//     printf("[mem] Array1D free %s\n", util::prettyBytes(self.tfn.colors.length * sizeof(float4)).c_str());
// #endif
  }
  if (self.tfn.colors.data) {
    CUDA_CHECK_NOEXCEPT(cudaDestroyTextureObject(self.tfn.colors.data));
    self.tfn.colors.data = { 0 };
  }
  if (self.tfn.colors.rawptr) {
    CUDA_CHECK_NOEXCEPT(cudaTrackedFree(self.tfn.colors.rawptr, self.tfn.colors.length * sizeof(float4)));
    self.tfn.colors.rawptr = nullptr;
//     util::total_n_bytes_allocated() -= self.tfn.colors.length * sizeof(float4);
// #ifdef VNR_VERBOSE_MEMORY_ALLOCS
//     printf("[mem] Linear free %s\n", util::prettyBytes(self.tfn.colors.length * sizeof(float4)).c_str());
// #endif
  }
  self.tfn.colors.length = 0;

  if (tfn_alpha_array_handler) {
    CUDA_CHECK_NOEXCEPT(cudaTrackedFreeArray(tfn_alpha_array_handler));
    tfn_color_array_handler = NULL;
//     util::total_n_bytes_allocated() -= self.tfn.alphas.length * sizeof(float);
// #ifdef VNR_VERBOSE_MEMORY_ALLOCS
//     printf("[mem] Array1D free %s\n", util::prettyBytes(self.tfn.alphas.length * sizeof(float)).c_str());
// #endif
  }
  if (self.tfn.alphas.data) {
    CUDA_CHECK_NOEXCEPT(cudaDestroyTextureObject(self.tfn.alphas.data));
    self.tfn.alphas.data = { 0 };
  }
  if (self.tfn.alphas.rawptr) {
    CUDA_CHECK_NOEXCEPT(cudaTrackedFree(self.tfn.alphas.rawptr, self.tfn.alphas.length * sizeof(float)));
    self.tfn.alphas.rawptr = nullptr;
//     util::total_n_bytes_allocated() -= self.tfn.alphas.length * sizeof(float);
// #ifdef VNR_VERBOSE_MEMORY_ALLOCS
//     printf("[mem] Linear free %s\n", util::prettyBytes(self.tfn.alphas.length * sizeof(float)).c_str());
// #endif
  }
  self.tfn.alphas.length = 0;
}

StructuredRegularVolume::StructuredRegularVolume()
{
  CreateSbtPtr(0); /* upload to GPU */
}

CUdeviceptr
StructuredRegularVolume::get_sbt_pointer(cudaStream_t stream)
{
  if (!GetSbtPtr()) {
    throw std::runtime_error("Volume device pointer not allocated");
  }
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
