#include "marching_cube.cuh"
#include "marching_cube_constants.cuh"

#include <api.h>
#include <api_internal.h>

#include "instantvnr_types.h"
#include "samplers/neural_sampler.h"
#include "networks/tcnn_device_api.h"

#include <vidi_highperformance_timer.h>

#include <cub/cub.cuh>

#include <iostream>
#include <fstream>

#define CACHE_VOXEL_VALUES 0

using namespace vnr;
constexpr int32_t n_threads_trilinear = 8;

struct VolumeInfoBase {
public:
  vec3i dims;
  vec3i dims_dual;
  float isovalue;

public:
  __device__ auto voxel_idx_to_coord(int64_t idx) const -> vec3i  {
    return vec3i(
       idx %  this->dims_dual[0],
      (idx /  this->dims_dual[0]) % this->dims_dual[1],
       idx / (this->dims_dual[0]  * this->dims_dual[1])
    );
  }

  __device__ auto lerp_verts(vec3i va, vec3i vb, float fa, float fb) const -> vec3f {
    float t = 0.0;
    if (abs(fa - fb) >= 0.001) {
          t = (this->isovalue - fa) / (fb - fa);
    }
    return vec3f(va) * (1 - t) + vec3f(vb) * t;
  }

  __device__ static auto vertex_offset(int32_t i) -> vec3i {
    return vec3i(INDEX_TO_VERTEX[3*i+0], INDEX_TO_VERTEX[3*i+1], INDEX_TO_VERTEX[3*i+2]);
  }
};


template<typename Impl = void>
struct VolumeDesc;

template<>
struct VolumeDesc<void> : public VolumeInfoBase {
public:
  float* data = nullptr;

public:
  VolumeDesc(float iso, vec3i dims, float* data) : VolumeInfoBase{dims, dims-1, iso}, data(data) {}

  template<typename K, typename... Types>
  void launch1D(K kernel, cudaStream_t stream, int32_t width, Types... args) const {
    constexpr int32_t block_size = n_threads_trilinear * n_threads_trilinear * n_threads_trilinear;
    const int32_t grid_size  = (int32_t)util::div_round_up<int64_t>(width, block_size);
    kernel<<<grid_size, block_size, 0, stream>>>(
      *this, width, args...
    );
  }

  template<typename K, typename... Types>
  void launch3D(K kernel, cudaStream_t stream, int32_t width, int32_t height, int32_t depth, Types... args) const {
    constexpr int32_t block_size = n_threads_trilinear * n_threads_trilinear * n_threads_trilinear;
    const int32_t grid_size  = (int32_t)util::div_round_up<int64_t>(width*height*depth, block_size);
    kernel<<<grid_size, block_size, 0, stream>>>(
      *this, width, height, depth, args...
    );
  }

  __device__ void compute_voxel_values(vec3i coord, float values[8]) const {
    // skip invalid coordinates
    if (gdt::any_greater_than_or_equal(coord, dims_dual)) return;
    // compute values
    #pragma unroll
    for (int32_t i = 0; i < 8; i++) {
      vec3i p = coord + vertex_offset(i);
      int64_t idx = p.x + int64_t(p.y) * this->dims[0] + p.z * int64_t(this->dims[0]) * this->dims[1];
      values[i] = data[idx];
    }
  }
};


template<typename Impl>
struct VolumeDesc : public VolumeInfoBase, private Impl {
public:
  VolumeDesc(float iso, vec3i dims, void* net) : VolumeInfoBase{dims, dims-1, iso}, Impl(net) {}

  template<typename K, typename... Types>
  void launch1D(K kernel, cudaStream_t stream, int32_t width, Types... args) const {
#if CACHE_VOXEL_VALUES
    constexpr int32_t block_size = n_threads_trilinear * n_threads_trilinear * n_threads_trilinear;
    const int32_t grid_size  = (int32_t)util::div_round_up<int64_t>(width, block_size);
    kernel<<<grid_size, block_size, 0, stream>>>(
      *this, width, args...
    );
#else
    Impl::launch_general(*this, kernel, stream, (uint32_t)width, width, args...);
#endif
  }

  template<typename K, typename... Types>
  void launch3D(K kernel, cudaStream_t stream, int32_t width, int32_t height, int32_t depth, Types... args) const {
    Impl::launch_general(*this, kernel, stream, (uint32_t)width*height*depth, width, height, depth, args...);
  }

  __device__ void compute_voxel_values(vec3i coord, float values[8]) const {
    for (int32_t i = 0; i < 8; i++) {
      vec3f p = vec3f(coord + vertex_offset(i)) / vec3f(dims);
      values[i] = (float)Impl::sample(p);
    }
  }
};


struct VoxelInfo {
  int64_t voxel_idx;
  uint8_t case_idx; // maximum MC_NUM_CASES
  uint8_t n_verts;  // maximum MC_CASE_ELEMENTS
#if CACHE_VOXEL_VALUES
  float values[8];
#endif
};


__device__ __forceinline__ uint64_t 
thread_index() {
  return (uint64_t)threadIdx.x + 
    (uint64_t)threadIdx.y * (uint64_t)blockDim.x + 
    (uint64_t)threadIdx.z * (uint64_t)blockDim.y * (uint64_t)blockDim.x + 
    (uint64_t)blockIdx.x * (uint64_t)blockDim.x * (uint64_t)blockDim.y * (uint64_t)blockDim.z;
}


template<typename VolumeInfo>
__global__ void kComputeActiveVoxels(
    const VolumeInfo volume, int32_t width, int32_t height, int32_t depth, 
    VoxelInfo* __restrict__ voxels, uint8_t* __restrict__ flags)
{
  const uint64_t index = thread_index();
  const vec3i coord = volume.voxel_idx_to_coord(index);

  assert(volume.dims_dual.x == width);
  assert(volume.dims_dual.y == height);
  assert(volume.dims_dual.z == depth);

  // We might have some workgroups run for voxels out of bounds due to the
  // padding to align to the workgroup size. We also only compute for voxels
  // on the dual grid, which has dimensions of volume_dims - 1
  // const bool invalid = gdt::any_greater_than_or_equal(coord, volume.dims_dual);
  const bool invalid = index >= (uint64_t)width * (uint64_t)height * (uint64_t)depth;

  // if (invalid) return; /* normal version */
  if (!block_any(!invalid)) return;

  VoxelInfo voxel;

  // Retrieve voxel values
#if CACHE_VOXEL_VALUES
  auto& values = voxel.values;
#else
  float values[8];
#endif
  volume.compute_voxel_values(coord, values);

  if (!invalid) 
  {
    // Compute the case this falls into to see if this voxel has vertices
    uint8_t case_idx = 0u;
    #pragma unroll
    for (auto i = 0u; i < 8u; i++) {
      if (values[i] <= volume.isovalue) case_idx |= 1u << i;
    }
    assert(case_idx < MC_NUM_CASES);

    // Compute the number of vertices
    uint8_t n_verts = 0u;
    #pragma unroll
    for (int8_t i = int8_t(0); MC_CASE_TABLE[case_idx * MC_CASE_ELEMENTS + i] != int8_t(-1); i++) {
      n_verts++;
    }
    assert(n_verts < MC_CASE_ELEMENTS);

    // Fill outputs
    voxel.voxel_idx = index;
    voxel.case_idx = case_idx;
    voxel.n_verts = n_verts;
    flags[index] = (n_verts > 0) ? uint8_t(1) : uint8_t(0);
    voxels[index] = voxel;
  }
}

template<typename VolumeInfo>
__global__ void kComputeVertices(
    const VolumeInfo volume, 
    const int64_t n_elements, 
    const VoxelInfo* __restrict__ voxels, 
    const int64_t* __restrict__ vertex_offsets,
    vec3f* __restrict__ vertex_positions)
{
  const uint64_t index = thread_index();
  
  // We might have some paddings, so we exclude invalid threads
  const bool invalid = (index >= n_elements);

  // if (invalid) return; /* normal version */
  if (!block_any(!invalid)) return;

  const VoxelInfo voxel = voxels[min(index, n_elements-1)];
  const vec3i coord = volume.voxel_idx_to_coord(voxel.voxel_idx);

  // Retrieve voxel values
#if CACHE_VOXEL_VALUES
  auto& values = voxel.values;
#else
  float values[8];
  volume.compute_voxel_values(coord, values);
#endif

  // Compute vertex positions
  if (!invalid) 
  {
    const uint32_t case_idx = voxel.case_idx;
    const int64_t vertex_offset = vertex_offsets[index];

    // Now we can finally compute and output the vertices
    for (int32_t i = 0u; MC_CASE_TABLE[case_idx * MC_CASE_ELEMENTS + i] != -1; i++) {
      auto edge = MC_CASE_TABLE[case_idx * MC_CASE_ELEMENTS + i];
      auto v0 = EDGE_VERTICES[2 * edge + 0];
      auto v1 = EDGE_VERTICES[2 * edge + 1];
      // Compute the interpolated vertex for this edge within the unit cell
      auto v = volume.lerp_verts(volume.vertex_offset(v0), volume.vertex_offset(v1), values[v0], values[v1]);
      // Offset the vertex into the global volume grid
      v = v + vec3f(coord) + 0.5f;
      vertex_positions[vertex_offset + i] = v;
    }
  }
}

template<typename T>
auto doExclusiveSum(int32_t num_items, const CUDABufferTyped<T> &input, CUDABufferTyped<T> &output) -> T
{
  const T *d_in = input.d_pointer(); 
  T *d_out = output.d_pointer();

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  std::size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items));

  // Allocate temporary storage
  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  // Run exclusive prefix sum
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Calculate the total sum
  T vLast, sLast;
  CUDA_CHECK(cudaMemcpy(&vLast, d_in  + num_items - 1, sizeof(vLast), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&sLast, d_out + num_items - 1, sizeof(sLast), cudaMemcpyDeviceToHost));

  // Cleanup
  CUDA_CHECK(cudaFree(d_temp_storage));
  return vLast + sLast;
}

template<typename T>
auto doStreamCompact(int32_t num_items, const CUDABufferTyped<T> &values, const CUDABufferTyped<uint8_t> &flags, CUDABufferTyped<T> &output, bool shrink = false) -> int32_t
{
  CUDABufferTyped<T> compacted;
  if (shrink) {
    compacted.alloc(num_items);
  }
  else {
    output.alloc(num_items); // TODO need to shrink it ...
    compacted.set_external(output);
  }

  // printf("doStreamCompact::allocate %s\n", util::prettyBytes(num_items * sizeof(T)).c_str());

  // Declare, allocate, and initialize device-accessible pointers for input, flags, and output
  const uint8_t *d_flags = flags.d_pointer();     // e.g., [1, 0, 0, 1, 0, 1, 1, 0]
  const T       *d_in    = values.d_pointer();    // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
  T             *d_out   = compacted.d_pointer(); // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
  int32_t       *d_num_selected_out = NULL;       // e.g., [ ]
  CUDA_CHECK(cudaMalloc(&d_num_selected_out, sizeof(int32_t)));

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  std::size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items));
  
  // Allocate temporary storage
  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  
  // Run selection
  CUDA_CHECK(cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items));
  CUDA_CHECK(cudaDeviceSynchronize());
  // d_out                 <-- [1, 4, 6, 7, 8, 5, 3, 2]
  // d_num_selected_out    <-- [4]
  CUDA_CHECK(cudaFree(d_temp_storage));
  
  // Return
  int32_t num_selected_out;
  CUDA_CHECK(cudaMemcpy(&num_selected_out, d_num_selected_out, sizeof(num_selected_out), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_num_selected_out));

  // Shrink
  if (shrink) {
    assert(!output.d_pointer());
    output.alloc(num_selected_out);
    CUDA_CHECK(cudaMemcpy(output.d_pointer(), compacted.d_pointer(), num_selected_out*sizeof(T), cudaMemcpyDeviceToDevice));

    // printf("doStreamCompact::allocate %s\n", util::prettyBytes(num_selected_out * sizeof(T)).c_str());
  }

  // printf("compact from %d to %d (%f%c)\n", num_items, num_selected_out, 100.f * (float)num_selected_out / num_items, '%');

  return num_selected_out;
}

template<typename VolumeInfo>
auto doComputeActiveVoxels(const VolumeInfo& volume, CUDABufferTyped<VoxelInfo>& active_voxels) -> int64_t
{
  /* 1. Compute active voxels
   * 2. Stream compact active voxel IDs
   *    - Scan is done on isActive buffer to get compaction offsets */ 
  const vec3i dims = volume.dims;
  const vec3i dims_dual = volume.dims_dual;
  const int64_t n_voxels = dims.long_product();
  const int64_t n_voxels_dual = dims_dual.long_product();

  CUDABufferTyped<VoxelInfo> voxels;
  voxels.alloc(n_voxels_dual);

  CUDABufferTyped<uint8_t> flags;
  flags.alloc(n_voxels_dual);

  // printf("doComputeActiveVoxels::allocate %s\n", util::prettyBytes(n_voxels_dual * (sizeof(VoxelInfo) + sizeof(uint8_t))).c_str());

  volume.launch3D(kComputeActiveVoxels<VolumeInfo>, (cudaStream_t)0, dims_dual.x, dims_dual.y, dims_dual.z, voxels.d_pointer(), flags.d_pointer());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Remove inactive voxels
  assert(n_voxels_dual < 0x7FFFFFFFULL);
  return doStreamCompact((int32_t)n_voxels_dual, voxels, flags, active_voxels, true);
}

template<typename VolumeInfo>
auto doComputeVertexOffsets(const VolumeInfo& volume, const int64_t n_voxels, const CUDABufferTyped<VoxelInfo>& voxels, CUDABufferTyped<int64_t>& vertex_offsets) -> int64_t
{
  CUDABufferTyped<int64_t> vertex_counts;
  vertex_counts.alloc(n_voxels);

  // printf("doComputeVertexOffsets::allocate %s\n", util::prettyBytes(n_voxels * sizeof(int64_t)).c_str());

  util::parallel_for_gpu(n_voxels, [v=voxels.d_pointer(), o=vertex_counts.d_pointer()] __device__ (int32_t i) {
    o[i] = (int64_t)v[i].n_verts;
  });

  const int64_t n_vertices = doExclusiveSum((int32_t)n_voxels, vertex_counts, vertex_offsets);

  return n_vertices;
}

template<typename VolumeInfo>
void doComputeVertices(
  const VolumeInfo& volume, const int64_t n_voxels, 
  const CUDABufferTyped<VoxelInfo>& voxels,
  const CUDABufferTyped<int64_t>& offsets,
  CUDABufferTyped<vec3f>& vertices)
{
  assert(n_voxels < 0x7FFFFFFFULL);
  volume.launch1D(kComputeVertices<VolumeInfo>, (cudaStream_t)0, (int32_t)n_voxels, voxels.d_pointer(), offsets.d_pointer(), vertices.d_pointer());
  CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename VolumeInfo>
void doMarchingCubeTemplate(const VolumeInfo& volume_info, CUDABufferTyped<vec3f>& vertices)
{
  vidi::details::HighPerformanceTimer timer;

  timer.start();

  /* Marching Cubes execution has 5 steps
   * 1. Compute active voxels
   * 2. Stream compact active voxel IDs
   *    - Scan is done on isActive buffer to get compaction offsets
   * 3. Compute # of vertices output by active voxels
   * 4. Scan # vertices buffer to produce vertex output offsets
   * 5. Compute and output vertices */

  CUDABufferTyped<VoxelInfo> active_voxels;
  CUDABufferTyped<int64_t> vertex_offsets;
  const int64_t n_active_voxels = doComputeActiveVoxels(volume_info, active_voxels);
  if (n_active_voxels > 0) {
    vertex_offsets.alloc(n_active_voxels);
    const int64_t n_vertices = doComputeVertexOffsets(volume_info, n_active_voxels, active_voxels, vertex_offsets);
    vertices.alloc(n_vertices);
    doComputeVertices(volume_info, n_active_voxels, active_voxels, vertex_offsets, vertices);
    vertex_offsets.free();
  }
  active_voxels.free();

  // Statistics
  timer.stop();
  const auto totaltime = timer.milliseconds();
  std::cout << "Marching Cube Time = "<< totaltime / 1000.0 << "s"<< std::endl;
}

/* network version */
template<int N_FEATURES_PER_LEVEL>
void doMarchingCubeTemplate__Network(const NeuralVolume& network, vec3i dims, float iso, CUDABufferTyped<vec3f>& vertices)
{
  int WIDTH = network.get_network_width();
  if (WIDTH == -1) {
    fprintf(stderr, "Incorrect MLP implementation for in-shader rendering (%s: line %d)\n", __FILE__, __LINE__);
    throw std::runtime_error("incorrect MLP implementation for in-shader rendering");
  }

  if (WIDTH == 16) {
    VolumeDesc<TcnnDeviceVolume<16,N_FEATURES_PER_LEVEL>> volume_info(iso, dims, network.get_network());
    return doMarchingCubeTemplate(volume_info, vertices);
  }

  if (WIDTH == 32) {
    VolumeDesc<TcnnDeviceVolume<32,N_FEATURES_PER_LEVEL>> volume_info(iso, dims, network.get_network());
    return doMarchingCubeTemplate(volume_info, vertices);
  }

  if (WIDTH == 64) {
    VolumeDesc<TcnnDeviceVolume<64,N_FEATURES_PER_LEVEL>> volume_info(iso, dims, network.get_network());
    return doMarchingCubeTemplate(volume_info, vertices);
  }

  fprintf(stderr, "Unsupported MLP WIDTH for in-shader rendering: %d (%s: line %d)\n", WIDTH, __FILE__, __LINE__);
  throw std::runtime_error("Unsupported MLP WIDTH for in-shader rendering");
}

void vnrMarchingCube(vnrVolume v, float iso, vnr::vec3f** ptr, size_t* size, bool cuda)
{
  CUDABufferTyped<vec3f> vertices;

  if (v->isNetwork()) {
    const auto ctx = std::dynamic_pointer_cast<NeuralVolumeContext>(v);
    const vec3i dims = ctx->desc.dims;

    const auto& network = ctx->neural;

    int N_FEATURES_PER_LEVEL = network.get_network_features_per_level();
    if (N_FEATURES_PER_LEVEL == -1) {
      fprintf(stderr, "Incorrect encoding method for in-shader rendering (%s: line %d)\n", __FILE__, __LINE__);
      throw std::runtime_error("Incorrect encoding method for in-shader rendering");
    }
  
    if      (N_FEATURES_PER_LEVEL == 1) doMarchingCubeTemplate__Network<1>(network, dims, iso, vertices);
    else if (N_FEATURES_PER_LEVEL == 2) doMarchingCubeTemplate__Network<2>(network, dims, iso, vertices);
    else if (N_FEATURES_PER_LEVEL == 4) doMarchingCubeTemplate__Network<4>(network, dims, iso, vertices);
    else if (N_FEATURES_PER_LEVEL == 8) doMarchingCubeTemplate__Network<8>(network, dims, iso, vertices);
    else throw std::runtime_error("expecting a simple volume");
  }

  else {
    const auto ctx = std::dynamic_pointer_cast<SimpleVolumeContext>(v);
    const vec3i dims = ctx->desc.dims;
    // printf("type = %d, dims = %d, %d, %d\n", (int)ctx->desc.type, dims.x, dims.y, dims.z);

    CUDABufferTyped<float> volume_data;
    {
      const int64_t n_voxels = dims.long_product();
      const auto* sampler = dynamic_cast<const StaticSampler*>(ctx->source.sampler.get_impl());
      volume_data.alloc_and_upload((float*)sampler->data(0), n_voxels);
    }

    VolumeDesc<void> volume_info(iso, dims, volume_data.d_pointer());

    doMarchingCubeTemplate(volume_info, vertices);
  }

  *size = vertices.size();
  if (!cuda) {
    *ptr = new vec3f[*size];
    vertices.download(*ptr, *size);
  }
  else {
    *ptr = vertices.release();
  }
}

void vnrSaveTriangles(std::string filename, const vnr::vec3f* ptr, size_t size)
{
  // Write output
  std::string str;
  for (int i = 0; i < size; ++i) {
    auto v = ptr[i];
    str += ("v " + std::to_string(v.x) + " " + std::to_string(v.y) + " " + std::to_string(v.z) + "\n");
  }
  for (int i = 0; i < size/3; ++i) {
    str += ("f " + std::to_string(3*i+1) + " " + std::to_string(3*i+2) + " " + std::to_string(3*i+3) + "\n");
  }

  std::ofstream outfile(filename);
  outfile.write(str.c_str(), str.length());
  outfile.close();

  return;
}
