#include "macrocell.h"

#ifndef MACROCELL_SIZE_MIP
#error MACROCELL_SIZE_MIP is not defined in the CMakeLists.txt
#endif

INSTANT_VNR_NAMESPACE_BEGIN

__device__ constexpr int MACROCELL_SIZE = 1 << MACROCELL_SIZE_MIP;

__forceinline__ __device__ void 
update_single_macrocell(const vec3i& voxel, const vec3i& dims, float* __restrict__ macrocells, float value)
{
  const vec3i cell = {
    voxel.x >> MACROCELL_SIZE_MIP,
    voxel.y >> MACROCELL_SIZE_MIP,
    voxel.z >> MACROCELL_SIZE_MIP,
  };

  if (cell.x < 0 || cell.x >= dims.x) return;
  if (cell.y < 0 || cell.y >= dims.y) return;
  if (cell.z < 0 || cell.z >= dims.z) return;

  assert(cell.x < dims.x);
  assert(cell.y < dims.y);
  assert(cell.z < dims.z);
  assert(cell.x >= 0);
  assert(cell.y >= 0);
  assert(cell.z >= 0);

  const uint32_t idx = cell.x + cell.y * dims.x + cell.z * dims.y * dims.x;
  float* __restrict__ vmin = macrocells + 2 * idx;
  float* __restrict__ vmax = macrocells + 2 * idx + 1;

  // All the value ranges are initialized as zero. Because all the values are within
  // range [0, 1], we can still compute a global min/max by adding a -1/+1 offset. 
  // We need to remove this offset when accessing value ranges.
  atomicMin(vmin, value - 1.f);
  atomicMax(vmax, value + 1.f);
}

__global__ void
update_macrocell_explicit(const uint32_t n_elements,
                          const vec3f* __restrict__ coords,
                          const float* __restrict__ values,
                          const vec3i dims,
                          const vec3i macrocell_dims,
                          float* __restrict__ macrocells)
{
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_elements)
    return;

  const vec3f coord = coords[i];
  const float value = values[i];

  const uint32_t x = clamp((uint32_t)floorf(coord.x * dims.x), uint32_t(0), uint32_t(dims.x-1));
  const uint32_t y = clamp((uint32_t)floorf(coord.y * dims.y), uint32_t(0), uint32_t(dims.y-1));
  const uint32_t z = clamp((uint32_t)floorf(coord.z * dims.z), uint32_t(0), uint32_t(dims.z-1));

  const int sx = (x % MACROCELL_SIZE) == 0 ? -1 : (x % MACROCELL_SIZE) == (MACROCELL_SIZE-1) ? 1 : 0;
  const int sy = (y % MACROCELL_SIZE) == 0 ? -1 : (y % MACROCELL_SIZE) == (MACROCELL_SIZE-1) ? 1 : 0;
  const int sz = (z % MACROCELL_SIZE) == 0 ? -1 : (z % MACROCELL_SIZE) == (MACROCELL_SIZE-1) ? 1 : 0;

  update_single_macrocell(vec3i(x,      y,      z     ), macrocell_dims, macrocells, value);
  update_single_macrocell(vec3i(x + sx, y,      z     ), macrocell_dims, macrocells, value);
  update_single_macrocell(vec3i(x,      y + sy, z     ), macrocell_dims, macrocells, value);
  update_single_macrocell(vec3i(x + sx, y + sy, z     ), macrocell_dims, macrocells, value);
  update_single_macrocell(vec3i(x,      y,      z + sz), macrocell_dims, macrocells, value);
  update_single_macrocell(vec3i(x + sx, y,      z + sz), macrocell_dims, macrocells, value);
  update_single_macrocell(vec3i(x,      y + sy, z + sz), macrocell_dims, macrocells, value);
  update_single_macrocell(vec3i(x + sx, y + sy, z + sz), macrocell_dims, macrocells, value);
}

__global__ void
update_macrocell_implicit(const uint32_t n_elements,
                          const uint64_t n_offset,
                          const vec3i dims,
                          const cudaTextureObject_t texture,
                          const vec3i macrocell_dims,
                          float* __restrict__ macrocells)
{
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_elements)
    return;

  const uint64_t idx = i + n_offset;
  const uint64_t stride = (uint64_t)dims.x * dims.y;

  const uint32_t x =  idx % dims.x;
  const uint32_t y = (idx % stride) / dims.x;
  const uint32_t z =  idx / stride;

  const float fx = (x + 0.5f) / dims.x;
  const float fy = (y + 0.5f) / dims.y;
  const float fz = (z + 0.5f) / dims.z;
  const float value = tex3D<float>(texture, fx, fy, fz);

  const int sx = (x % MACROCELL_SIZE) == 0 ? -1 : (x % MACROCELL_SIZE) == (MACROCELL_SIZE-1) ? 1 : 0;
  const int sy = (y % MACROCELL_SIZE) == 0 ? -1 : (y % MACROCELL_SIZE) == (MACROCELL_SIZE-1) ? 1 : 0;
  const int sz = (z % MACROCELL_SIZE) == 0 ? -1 : (z % MACROCELL_SIZE) == (MACROCELL_SIZE-1) ? 1 : 0;

  update_single_macrocell(vec3i(x,      y,      z     ), macrocell_dims, macrocells, value);
  update_single_macrocell(vec3i(x + sx, y,      z     ), macrocell_dims, macrocells, value);
  update_single_macrocell(vec3i(x,      y + sy, z     ), macrocell_dims, macrocells, value);
  update_single_macrocell(vec3i(x + sx, y + sy, z     ), macrocell_dims, macrocells, value);
  update_single_macrocell(vec3i(x,      y,      z + sz), macrocell_dims, macrocells, value);
  update_single_macrocell(vec3i(x + sx, y,      z + sz), macrocell_dims, macrocells, value);
  update_single_macrocell(vec3i(x,      y + sy, z + sz), macrocell_dims, macrocells, value);
  update_single_macrocell(vec3i(x + sx, y + sy, z + sz), macrocell_dims, macrocells, value);
}

__global__ void // computing macrocell value range offline
macrocell_value_range_kernel(const uint32_t mcDimsX,
                             const uint32_t mcDimsY,
                             const uint32_t mcDimsZ,
                             const uint32_t mcWidth,
                             vec2f* __restrict__ mcData,
                             const vec3i volumeDims,
                             cudaTextureObject_t volumeTexture)
{
  // 3D kernel launch
  vec3i mcID(threadIdx.x+blockIdx.x*blockDim.x,
             threadIdx.y+blockIdx.y*blockDim.y,
             threadIdx.z+blockIdx.z*blockDim.z);

  if (mcID.x >= mcDimsX) return;
  if (mcID.y >= mcDimsY) return;
  if (mcID.z >= mcDimsZ) return;

  int mcIdx = mcID.x + mcDimsX*(mcID.y + mcDimsY*mcID.z);
  vec2f &mc = mcData[mcIdx];

  // compute begin/end of VOXELS for this macro-cell
  vec3i begin = max(mcID  * vec3i(mcWidth) - 1, vec3i(0));
  vec3i end   = min(begin + vec3i(mcWidth) + /* plus one for tri-lerp!*/ 2, volumeDims);

  range1f valueRange;
  for (int iz = begin.z; iz < end.z; iz++)
    for (int iy = begin.y; iy < end.y; iy++)
      for (int ix = begin.x; ix < end.x; ix++) {
          float f;
          tex3D(&f, volumeTexture, 
                (ix + 0.5f) / volumeDims.x, 
                (iy + 0.5f) / volumeDims.y, 
                (iz + 0.5f) / volumeDims.z);
          valueRange.extend(f);
        }
  mc.x = valueRange.lo - 1.f;
  mc.y = valueRange.hi + 1.f;
}

__global__ void // compute macrocell opacity all together
macrocell_max_opacity_kernel(const uint32_t num_cells, 
                             const DeviceTransferFunction tfn, 
                             const vec2f* __restrict__ cell_value_range, 
                             float* __restrict__ cell_max_opacity)
{
  extern __shared__ float shared_alphas[];

  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  assert(blockDim.x == tfn.alphas.length);

  // load tfn into shared memory (assume the number of threads per group equals the length of the alpha array)
  shared_alphas[threadIdx.x] = ((float*)tfn.alphas.rawptr)[threadIdx.x];
  __syncthreads();
  const float* __restrict__ alphas = shared_alphas;

  // access macrocell value range
  if (i >= num_cells) return;
  auto range = cell_value_range[i];
  range.x += 1.f;
  range.y -= 1.f; // see function: update_single_macrocell

  // compute the max opacity for the cell
  assert(tfn.alphas.length > 0); // for the first frame, tfn.alphas.length might be zero

  const auto lower = (clamp(range.x, tfn.range.lower, tfn.range.upper) - tfn.range.lower) * tfn.range_rcp_norm;
  const auto upper = (clamp(range.y, tfn.range.lower, tfn.range.upper) - tfn.range.lower) * tfn.range_rcp_norm;
  uint32_t i_lower = floorf(fmaf(lower, float(tfn.alphas.length-1), 0.5f)) - 1;
  uint32_t i_upper = floorf(fmaf(upper, float(tfn.alphas.length-1), 0.5f)) + 1;
  i_lower = clamp<uint32_t>(i_lower, 0, tfn.alphas.length-1);
  i_upper = clamp<uint32_t>(i_upper, 0, tfn.alphas.length-1);

  assert(i_lower < tfn.alphas.length);
  assert(i_upper < tfn.alphas.length);

  float opacity = 0.f;
  for (auto i = i_lower; i <= i_upper; ++i) {
    opacity = std::max(opacity, alphas[i]);
  }
  cell_max_opacity[i] = opacity;
}

void MacroCell::set_shape(vec3i dims)
{
  m_volume_dims = dims;

  m_dims = util::div_round_up(dims, vec3i(MACROCELL_SIZE));
  m_spacings = vec3f(MACROCELL_SIZE) / vec3f(dims);
}

void MacroCell::set_external(MacroCell& external)
{
  m_volume_dims = external.m_volume_dims;
  m_dims = external.m_dims;
  m_spacings = external.m_spacings;
  m_max_opacity_buffer.set_external(external.m_max_opacity_buffer);
  m_value_range_buffer.set_external(external.m_value_range_buffer);
  m_is_external = true;
}

void MacroCell::allocate()
{
  m_value_range_buffer.resize(m_dims.long_product() * sizeof(range1f), nullptr);
  m_value_range_buffer.memset(0, nullptr);
  m_max_opacity_buffer.resize(m_dims.long_product() * sizeof(float), nullptr);
  m_is_external = false;
}

void MacroCell::compute_everything(cudaTextureObject_t volume)
{
  for (int z = 0; z < m_volume_dims.z; ++z) {
    util::linear_kernel(update_macrocell_implicit, 0, 0, 
                        (uint32_t)m_volume_dims.x * m_volume_dims.y, 
                        (uint64_t)m_volume_dims.x * m_volume_dims.y * z, 
                        m_volume_dims, volume, m_dims,
                        (float*)d_value_range());
  }
  // util::trilinear_kernel(macrocell_value_range_kernel, 0, 0, 
  //     m_dims.x, m_dims.y, m_dims.z, 
  //     MACROCELL_SIZE, (vec2f*)d_value_range(), 
  //     m_volume_dims, volume);
}

void MacroCell::update_explicit(vec3f* d_coords, float* d_values, size_t count, cudaStream_t stream)
{
  util::linear_kernel(update_macrocell_explicit, 0, stream, (uint32_t)count, 
                      d_coords, d_values, m_volume_dims, m_dims, 
                      (float*)d_value_range());
}

void MacroCell::update_max_opacity(const DeviceTransferFunction& tfn, cudaStream_t stream) 
{
  if (tfn.alphas.length <= 0) return;

  const size_t shmem = tfn.alphas.length * sizeof(float);
  const size_t n_elements = m_dims.long_product();

  macrocell_max_opacity_kernel<<<(uint32_t)util::div_round_up(n_elements, tfn.alphas.length), (uint32_t)tfn.alphas.length, (uint32_t)shmem, stream>>>(
    (uint32_t)n_elements, tfn, (vec2f*)d_value_range(), (float*)d_max_opacity()
  );
}

INSTANT_VNR_NAMESPACE_END
