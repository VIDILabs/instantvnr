#include "neural_sampler.h"

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
} // namespace vidi
#define VIDI_VOLUME_EXTERNAL_TYPE_ENUM
#include <vidi_parallel_algorithm.h>
#include <vidi_volume_reader.h>


#ifdef ENABLE_OUT_OF_CORE

#if !defined(_WIN32)
#define AIO_LINUX
#include <libaio.h>
#endif
#ifdef AIO_INTEL
#include <aio.h>
typedef struct aiocb aiocb_t;
#endif
#endif // ENABLE_OUT_OF_CORE

#ifdef ENABLE_OPENVKL
#include <openvkl/openvkl.h>
#include <openvkl_testing.h>
#endif

#include <tbb/parallel_for.h>

#include <random>
#include <iostream>
#include <string>
#include <vector>
#include <deque>


#ifdef ENABLE_LOGGING
#define log() std::cout
#else
static std::ostream null_output_stream(0);
#define log() null_output_stream
#endif

static std::mt19937 rng;

static uint32_t random_uint32(const uint32_t& min, const uint32_t& max) {
  ASSERT_THROW(min < max, "calling 'random_uint32' with invalid range.");
  std::uniform_int_distribution<uint32_t> distribution(min, max);
  return distribution(rng);
}

static uint32_t random_uint32(const uint32_t& count) {
  ASSERT_THROW(count != 0, "calling 'random_uint32' with zero range.");
  return random_uint32(0, count - 1);
}

static uint64_t random_uint64(const uint64_t& min, const uint64_t& max) {
  ASSERT_THROW(min < max, "calling 'uint64_random' with invalid range.");
  std::uniform_int_distribution<uint64_t> distribution(min, max);
  return distribution(rng);
}

static uint64_t random_uint64(const uint64_t& count) {
  ASSERT_THROW(count != 0, "calling 'uint64_random' with zero range.");
  return random_uint64(0, count - 1);
}

static float random_uniform(const float& min, const float& max) {
  std::uniform_real_distribution<float> distribution(min, max);
  return distribution(rng);
}

namespace vnr {

void random_hbuffer_uniform(float* d_buffer, float* h_buffer, size_t batch) {
#if USE_GPU_RANDOM_NUMBER_GENERSTOR
  random_dbuffer_uniform(d_buffer, batch, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(h_buffer, d_buffer, batch * sizeof(float), cudaMemcpyDeviceToHost, nullptr));
#else
  random_hbuffer_uniform(h_buffer, batch);
#endif
}

void random_hbuffer_uniform(float* h_buffer, size_t batch) {
#if USE_GPU_RANDOM_NUMBER_GENERSTOR
  static CUDABuffer buffer;
  buffer.resize(batch * sizeof(float), nullptr);
  random_dbuffer_uniform((float*)buffer.d_pointer(), batch, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(h_buffer, (float*)buffer.d_pointer(), buffer.sizeInBytes, cudaMemcpyDeviceToHost, nullptr));
#else
  for (size_t i = 0; i < batch; ++i) h_buffer[i] = random_uniform(0.f, 1.f);
#endif
}

void random_hbuffer_uint32(uint32_t* h_buffer, size_t batch, uint32_t min, uint32_t max)
{
#if USE_GPU_RANDOM_NUMBER_GENERSTOR
  static CUDABuffer buffer;
  buffer.resize(batch * sizeof(uint32_t), nullptr);
  random_dbuffer_uint32((uint32_t*)buffer.d_pointer(), batch, min, max, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(h_buffer, (uint32_t*)buffer.d_pointer(), buffer.sizeInBytes, cudaMemcpyDeviceToHost, nullptr));
#else
  for (size_t i = 0; i < batch; ++i) h_buffer[i] = random_uint32(min, max);
#endif
}

void random_hbuffer_uint64(uint64_t* h_buffer, size_t batch, uint64_t min, uint64_t max)
{
#if USE_GPU_RANDOM_NUMBER_GENERSTOR
  static CUDABuffer buffer;
  buffer.resize(batch * sizeof(uint64_t), nullptr);
  random_dbuffer_uint64((uint64_t*)buffer.d_pointer(), batch, min, max, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(h_buffer, (uint64_t*)buffer.d_pointer(), buffer.sizeInBytes, cudaMemcpyDeviceToHost, nullptr));
#else
  for (size_t i = 0; i < batch; ++i) h_buffer[i] = random_uint64(min, max);
#endif
}

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

template <typename T>
static T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

template <typename T>
static T next_multiple(T val, T divisor) {
	return div_round_up(val, divisor) * divisor;
}

static inline float
read_typed_pointer(char* buffer, size_t id, ValueType type)
{
  switch (type) {
  case VALUE_TYPE_UINT8:  return ((uint8_t* )buffer)[id];
  case VALUE_TYPE_INT8:   return ((int8_t*  )buffer)[id];
  case VALUE_TYPE_UINT16: return ((uint16_t*)buffer)[id];
  case VALUE_TYPE_INT16:  return ((int16_t* )buffer)[id]; 
  case VALUE_TYPE_UINT32: return ((uint32_t*)buffer)[id];
  case VALUE_TYPE_INT32:  return ((int32_t* )buffer)[id]; 
  case VALUE_TYPE_FLOAT:  return ((float*   )buffer)[id];   
  case VALUE_TYPE_DOUBLE: return ((double*  )buffer)[id];  
  default: throw std::runtime_error("unknown data type");
  }
}

static inline bool
isclose(float a, float b, float rel_tol = 10*float_epsilon, float abs_tol = 0.0)
{
  return std::abs(a - b) <= std::max(rel_tol * std::max(std::abs(a), std::abs(b)), abs_tol);
}

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

namespace {

// template<class T>
// constexpr const T& clamp(const T& v, const T& lo, const T& hi)
// {
//     return (v < lo) ? lo : (hi < v) ? hi : v;
// }

template<typename IType>
std::shared_ptr<char[]>
convert_volume(std::shared_ptr<char[]> idata, size_t size, float vmin, float vmax)
{
  std::shared_ptr<char[]> odata;
  odata.reset(new char[size * sizeof(float)]);

  tbb::parallel_for(size_t(0), size, [&](size_t idx) {
    auto* i = (IType*)&idata[idx * sizeof(IType)];
    auto* o = (float*)&odata[idx * sizeof(float)];
#ifdef TEST_SIREN
    *o = clamp((static_cast<float>(*i) - (float)vmin) / ((float)vmax - (float)vmin), 0.f, 1.f) * 2.f - 1.f;
#else
    *o = clamp((static_cast<float>(*i) - (float)vmin) / ((float)vmax - (float)vmin), 0.f, 1.f);
#endif
  });

  return odata;
}

template<>
std::shared_ptr<char[]>
convert_volume<float>(std::shared_ptr<char[]> idata, size_t size, float vmin, float vmax)
{
  tbb::parallel_for(size_t(0), size, [&](size_t idx) {
    auto* i = (float*)&idata[idx * sizeof(float)];
#ifdef TEST_SIREN
    *i = clamp((static_cast<float>(*i) - (float)vmin) / ((float)vmax - (float)vmin), 0.f, 1.f) * 2.f - 1.f;
#else
    *i = clamp((static_cast<float>(*i) - (float)vmin) / ((float)vmax - (float)vmin), 0.f, 1.f);
#endif
  });

  return idata;
}

template<typename T>
static range1f
compute_scalar_fminmax(const void* _array, size_t count)
{
  using vidi::parallel::compute_scalar_minmax;
  auto r = compute_scalar_minmax<T>(_array, count, 0);
  return range1f((float)r.first, (float)r.second);
}

} // namespace

void
StaticSampler::load(const MultiVolume::File& desc,
                    vec3i dims, dtype type, range1f minmax,
                    std::shared_ptr<char[]>& buffer,
                    range1f& value_range_unnormalized,
                    range1f& value_range_normalized)
{
  const auto& offset = desc.offset;
  const auto& filename = desc.filename;
  const auto& is_big_endian = desc.bigendian;

  /* load data from file */
  {
    vidi::StructuredRegularVolumeDesc desc;
    desc.dims.x = dims.x;
    desc.dims.y = dims.y;
    desc.dims.z = dims.z;
    desc.type = (vidi::VoxelType)type;
    desc.offset = offset;
    desc.is_big_endian = is_big_endian;
    buffer = vidi::read_volume_structured_regular(filename, desc);
  }

  /* copy data to GPU */
  const size_t count = (size_t)dims.x * dims.y * dims.z;

  /* convert volume into floats */
  range1f range;
  {
    if (minmax.is_empty()) {
      switch (type) {
      case VALUE_TYPE_UINT8: range = compute_scalar_fminmax<uint8_t>(buffer.get(), count); break;
      case VALUE_TYPE_INT8: range = compute_scalar_fminmax<int8_t>(buffer.get(), count); break;
      case VALUE_TYPE_UINT16: range = compute_scalar_fminmax<uint16_t>(buffer.get(), count); break;
      case VALUE_TYPE_INT16: range = compute_scalar_fminmax<int16_t>(buffer.get(), count); break;
      case VALUE_TYPE_UINT32: range = compute_scalar_fminmax<uint32_t>(buffer.get(), count); break;
      case VALUE_TYPE_INT32: range = compute_scalar_fminmax<int32_t>(buffer.get(), count); break;
      case VALUE_TYPE_FLOAT: range = compute_scalar_fminmax<float>(buffer.get(), count); break;
      case VALUE_TYPE_DOUBLE: range = compute_scalar_fminmax<double>(buffer.get(), count); break;
      default: throw std::runtime_error("unknown data type");
      }
    }
    else {
      range = minmax;
    }

    switch (type) {
    case VALUE_TYPE_UINT8: buffer = convert_volume<uint8_t>  (buffer, count, range.lower, range.upper); break;
    case VALUE_TYPE_INT8: buffer = convert_volume<int8_t>    (buffer, count, range.lower, range.upper); break;
    case VALUE_TYPE_UINT16: buffer = convert_volume<uint16_t>(buffer, count, range.lower, range.upper); break;
    case VALUE_TYPE_INT16: buffer = convert_volume<int16_t>  (buffer, count, range.lower, range.upper); break;
    case VALUE_TYPE_UINT32: buffer = convert_volume<uint32_t>(buffer, count, range.lower, range.upper); break;
    case VALUE_TYPE_INT32: buffer = convert_volume<int32_t>  (buffer, count, range.lower, range.upper); break;
    case VALUE_TYPE_FLOAT: buffer = convert_volume<float>    (buffer, count, range.lower, range.upper); break;
    case VALUE_TYPE_DOUBLE: buffer = convert_volume<double>  (buffer, count, range.lower, range.upper); break;
    default: throw std::runtime_error("unknown data type");
    }
  }
  value_range_unnormalized = range;
  value_range_normalized.lower = 0.f;
  value_range_normalized.upper = 1.f;
  // std::tie(value_range_normalized.lower, value_range_normalized.upper) = vidi::parallel::compute_scalar_minmax<float>(buffer.get(), count, 0);

  log() << "[vnr] unnormalized range " << value_range_unnormalized.lower << " " << value_range_unnormalized.upper << std::endl;
  log() << "[vnr] normalized range " << value_range_normalized.lower << " " << value_range_normalized.upper << std::endl;
}

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

template<typename F>
static float
nearest_vkl(const vec3f& p_cell_centered, const vec3i& m_dims, const F& accessor)
{
  const vec3i vidx = vec3i(p_cell_centered);
  return accessor(vidx);
}

template<typename F>
static float 
trilinear_vkl(const vec3f& p_cell_centered, const vec3i& m_dims, const F& accessor)
{
  const vec3f pb = p_cell_centered - 0.5f;
  vec3f w, iw; {
    w.x = std::modf(pb.x, &iw.x);
    w.y = std::modf(pb.y, &iw.y);
    w.z = std::modf(pb.z, &iw.z);
    assert(iw.x == std::floor(pb.x));
    assert(iw.y == std::floor(pb.y));
    assert(iw.z == std::floor(pb.z));
  }
  const vec3i vidx_0 = clamp(vec3i(iw),  vec3i(0), m_dims - 1);
  const vec3i vidx_1 = clamp(vidx_0 + 1, vec3i(0), m_dims - 1);
  float c000 = accessor(vidx_0);
  float c001 = accessor(vec3i(vidx_1.x, vidx_0.y, vidx_0.z));
  float c010 = accessor(vec3i(vidx_0.x, vidx_1.y, vidx_0.z));
  float c011 = accessor(vec3i(vidx_1.x, vidx_1.y, vidx_0.z));
  float c100 = accessor(vec3i(vidx_0.x, vidx_0.y, vidx_1.z));
  float c101 = accessor(vec3i(vidx_1.x, vidx_0.y, vidx_1.z));
  float c110 = accessor(vec3i(vidx_0.x, vidx_1.y, vidx_1.z));
  float c111 = accessor(vidx_1);
  return (1-w.x)*(1-w.y)*(1-w.z)*c000 + w.x*(1-w.y)*(1-w.z)*c001
        +(1-w.x)*   w.y *(1-w.z)*c010 + w.x*   w.y *(1-w.z)*c011
        +(1-w.x)*(1-w.y)*   w.z *c100 + w.x*(1-w.y)*   w.z *c101
        +(1-w.x)*   w.y *   w.z *c110 + w.x*   w.y *   w.z *c111;
}

/////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////

static vec3i
to_grid_index(size_t index, vec3i grid)
{
  const auto stride_y = (size_t)grid.x;
  const auto stride_z = (size_t)grid.y * (size_t)grid.x;
  return vec3i(index % stride_y, (index % stride_z) / stride_y, index / stride_z);
}

static size_t
flatten_grid_index(vec3i index, vec3i grid)
{
  const auto stride_y = (size_t)grid.x;
  const auto stride_z = (size_t)grid.y * (size_t)grid.x;
  return (size_t)index.x + (size_t)index.y * stride_y + (size_t)index.z * stride_z;
}

static vec3i
random_grid_index(vec3i grid)
{
  const auto index = random_uint64(grid.long_product());
  return to_grid_index(index, grid);
}

static bool
is_in_grid(vec3i index, vec3i grid)
{
  return index.x >= 0 && index.y >= 0 && index.z >= 0 && index.x < grid.x && index.y < grid.y && index.z < grid.z;
}

static bool
is_in_bounds(vec3i index, box3i bounds)
{
  return !(gdt::any_less_than(index,bounds.lower) || gdt::any_greater_than_or_equal(index,bounds.upper));
}

#ifdef AIO_INTEL
static volatile int aio_flg = 0;
static void aio_CompletionRoutine(sigval_t sigval) {
  aio_flg = 1;
}
#endif

template<uint64_t STREAM_SIZE>
struct StreamLoader {
private:
  // TODO: this should be optimized by using a delicated I/O thread to maintain the random buffer. Then,
  // there will be two queues, one "finished" queue and one "busy" queue. The "finished" queue contains
  // buffer that are recently updated, and can be used for sampling. The "busy" queue should contain I/O
  // jobs that are running. Once a job in the "busy" queue is done, it should be atomically swapped with
  // a queue in the "finished" queue.
#ifdef AIO_LINUX
  io_context_t aio_ctx;
  uint64_t aio_job_count = 0;
#else
#ifdef AIO_INTEL
  std::vector<aiocb_t>  aio_jobs_data;
  std::vector<aiocb_t*> aio_jobs_list;
  uint64_t aio_job_count = 0;
  struct sigevent sig;
#else
  std::deque<vidi::future_buffer_t> aio_jobs;
#endif
#endif 

  uint64_t n_streams;

public:
  void init(uint64_t n_concurrent_streams) 
  {
#ifdef AIO_LINUX
    // create a I/O context
    memset(&aio_ctx, 0, sizeof(aio_ctx));
    if (io_setup(n_concurrent_streams, &aio_ctx) != 0) {
      throw std::runtime_error("[aio] error when calling io_setup");
    }
#else
#ifdef AIO_INTEL
    aio_jobs_data.resize(n_concurrent_streams);
    aio_jobs_list.resize(n_concurrent_streams);
    for (uint64_t i = 0; i < n_concurrent_streams; ++i) {
      aio_jobs_list[i] = &aio_jobs_data[i];
      aio_jobs_data[i].aio_lio_opcode = LIO_READ;
    }
    sig.sigev_notify          = SIGEV_THREAD;
    sig.sigev_notify_function = aio_CompletionRoutine;
#endif
#endif
    n_streams = n_concurrent_streams;
  }

  void request_buffer(vidi::FileMap reader, char* data, uint64_t offset, uint64_t nbytes)
  {
    for (uint64_t read = 0; read < nbytes; read += STREAM_SIZE)
      request_stream(reader, data + read, offset + read, min(STREAM_SIZE, nbytes - read));
  }

  void request_stream(vidi::FileMap reader, char* data, uint64_t offset, uint64_t nbytes) 
  {
#ifdef AIO_LINUX
    struct iocb cb;
    auto* ptr = &cb;
    io_prep_pread(ptr, reader->fd, data, nbytes, offset);
    if (io_submit(aio_ctx, 1, &ptr) != 1) {
      throw std::runtime_error("[aio] io_submit");
    }
    ++aio_job_count;
#else
#ifdef AIO_INTEL
    auto& cb = aio_jobs_data[aio_job_count];
    cb.aio_fildes = reader->fd;
    cb.aio_buf    = data;
    cb.aio_nbytes = nbytes;
    cb.aio_offset = (intptr_t)offset;
    cb.aio_sigevent.sigev_notify = SIGEV_NONE;
    ++aio_job_count;
#else
    aio_jobs.emplace_back(vidi::filemap_random_read_async(reader, offset, data, nbytes));
    assert(aio_jobs.size() <= n_streams);
#endif
#endif
  }

  void run_for_all() {
#ifdef AIO_INTEL
    if (lio_listio(LIO_WAIT, aio_jobs_list.data(), aio_job_count, 0)) 
      throw std::runtime_error("[aio] lio_listio");
#endif
  }

  void wait_for_all() {
#ifdef AIO_LINUX
    for (uint64_t i = 0; i < aio_job_count; ++i) {
      struct io_event event;
      if (int err = io_getevents(aio_ctx, 1, 1, &event, NULL) != 1) {
        throw std::runtime_error("[aio] io_getevents");
      }
    }
    aio_job_count = 0;
#else
#ifdef AIO_INTEL
    // while (!aio_flg); 
    // aio_flg = 0;
    aio_job_count = 0;
#else
    while (!aio_jobs.empty()) {
      aio_jobs.front()->wait();
      aio_jobs.pop_front();
    }
#endif
#endif
  }
};

struct RandomBuffer
{
  constexpr static uint64_t STREAM_SIZE = 32*1024;
  constexpr static uint64_t ALIGNMENT = 512;

  const uint64_t NUM_CONCURRENT_BLOCKS;
  const uint64_t NUM_BLOCKS;

  struct Block {
    vec3i index{};
    uint64_t offset{};
    uint64_t length{};
    box3i bounds{};
    box3i bounds_with_ghost{};
  };

  std::vector<Block> blocks;

private:
  uint64_t element_size; // in bytes
  
  vec3i block_index_space;
  vec3i block_dims;
  vec3i block_dims_with_ghosts;
  uint64_t block_size_aligned; // in bytes

  uint64_t num_jobs_per_slice;
  char* block_data{ nullptr };

  StreamLoader<STREAM_SIZE> streamer;

  const vidi::FileMap freader;
  const uint64_t fsize{ 0 };
  const vec3i fdims{ 0 };

public:
  RandomBuffer(vidi::FileMap reader, vec3i dims, uint64_t element_size, const uint64_t NUM_CONCURRENT_BLOCKS = 1024, const uint64_t NUM_BLOCKS = 1024*64) 
    : NUM_CONCURRENT_BLOCKS(NUM_CONCURRENT_BLOCKS)
    , NUM_BLOCKS(NUM_BLOCKS)
    , element_size(element_size)
    , freader(reader)
    , fsize(reader->getFileSize())
    , fdims(dims)
  {
    log() << "[RandomBuffer] # concurrent blocks = " << NUM_CONCURRENT_BLOCKS << ", # total blocks = " << NUM_BLOCKS << "\n";

    ASSERT_THROW(STREAM_SIZE % element_size == 0, "[aio] bad stream alignment");
    {
      block_dims.x = fdims.x;
      block_dims.y = min(div_round_up(STREAM_SIZE, (uint64_t)fdims.x*element_size), (uint64_t)fdims.y);
      block_dims.z = min(1, fdims.z);

      block_dims_with_ghosts.x = block_dims.x;
      block_dims_with_ghosts.y = min(block_dims.y + 2, fdims.y);
      block_dims_with_ghosts.z = min(block_dims.z + 2, fdims.z);

      block_index_space.x = 1;
      block_index_space.y = div_round_up(fdims.y, block_dims.y);
      block_index_space.z = div_round_up(fdims.z, block_dims.z);
    }

    // maximum possible jobs
    ASSERT_THROW(block_dims_with_ghosts.x*block_dims_with_ghosts.y*element_size > STREAM_SIZE, "[aio] incorrect");
    num_jobs_per_slice = div_round_up(block_dims_with_ghosts.x*block_dims_with_ghosts.y*element_size, STREAM_SIZE);
    streamer.init(next_multiple(NUM_CONCURRENT_BLOCKS * num_jobs_per_slice * block_dims_with_ghosts.z, (uint64_t)16));

    // allocate data (with alignment)
    blocks.resize(NUM_BLOCKS);
    block_size_aligned = next_multiple(block_dims_with_ghosts.long_product() * element_size, ALIGNMENT);
#if defined(_WIN32) // aligned_alloc is not implemented in MSVC
    block_data = (char*)_aligned_malloc(NUM_BLOCKS * block_size_aligned, ALIGNMENT);
#else
    block_data = (char*)aligned_alloc(ALIGNMENT, NUM_BLOCKS * block_size_aligned);
#endif

    log() << "[aio] preloading " << NUM_BLOCKS << " blocks" << std::endl;
    for (int i = 0; i < NUM_BLOCKS; i += NUM_CONCURRENT_BLOCKS) {
      submit_all_jobs(i);
      wait_all_jobs();
    }

    submit_all_jobs();
  }

  ~RandomBuffer()
  {
#if defined(_WIN32)
    if (block_data) _aligned_free(block_data);
#else
    if (block_data) free(block_data);
#endif
  }

  // number i is the job index, index is the block index in file
  void submit_one_job(uint64_t i, vec3i block_index)
  {
    assert(i < NUM_BLOCKS);

    ASSERT_THROW(is_in_grid(block_index, block_index_space), "[aio] invalid block index");

    const vec3i voxel_0 = block_index * block_dims;
    const vec3i voxel_1 = min(voxel_0 + block_dims, fdims);
    ASSERT_THROW(is_in_grid(voxel_0, fdims),     "[aio] incorrect");
    ASSERT_THROW(is_in_grid(voxel_1 - 1, fdims), "[aio] incorrect");
    ASSERT_THROW(voxel_0.x == 0,       "[aio] violate the design");
    ASSERT_THROW(voxel_1.x == fdims.x, "[aio] violate the design");

    Block block;
    block.index = block_index;
    block.bounds = box3i(voxel_0, voxel_1);
    block.offset = flatten_grid_index(voxel_0, fdims);
    block.length = (voxel_1 - voxel_0).long_product();
    ASSERT_THROW(block.length > 0, "[aio] zero block");

    const vec3i voxel_0_ghost = max(voxel_0 - vec3i(1), vec3i(0));
    const vec3i voxel_1_ghost = min(voxel_1 + vec3i(1), fdims);
    block.bounds_with_ghost = box3i(voxel_0_ghost, voxel_1_ghost);
    ASSERT_THROW(is_in_grid(voxel_0_ghost, fdims), "[aio] incorrect");
    ASSERT_THROW(voxel_0_ghost.x == 0,       "[aio] violate the design");
    ASSERT_THROW(voxel_1_ghost.x == fdims.x, "[aio] violate the design");
    ASSERT_THROW(block.bounds_with_ghost.size().long_product()*element_size <= block_size_aligned, "[aio] invalid block");

    blocks[i] = block;

    const vec3i ghost_dims = (voxel_1_ghost - voxel_0_ghost);
    ASSERT_THROW(ghost_dims.long_product() <= block_size_aligned, "[aio] invalid ghost block");

    auto* data = block_data + i * block_size_aligned;
    ASSERT_THROW(uint64_t(data) % ALIGNMENT == 0, "[aio] block data not aligned");

    const vec3i& block_begin = voxel_0_ghost;

    for (int z = voxel_0_ghost.z; z < voxel_1_ghost.z; ++z) {
      const vec3i slice_begin = vec3i(voxel_0_ghost.x, voxel_0_ghost.y, z);
      ASSERT_THROW(is_in_grid(slice_begin, fdims), "[aio] incorrect");
      ASSERT_THROW(is_in_grid(slice_begin - block_begin, ghost_dims), "[aio] incorrect");

      const uint64_t slice_offset_b = flatten_grid_index(slice_begin - block_begin, ghost_dims) * element_size;
      const uint64_t slice_offset_f = flatten_grid_index(slice_begin, fdims) * element_size;

      const uint64_t nbytes_to_read = (uint64_t)ghost_dims.x * ghost_dims.y * element_size;
      ASSERT_THROW(nbytes_to_read > 0, "[aio] invalid block nbytes");

      ASSERT_THROW(slice_offset_b + nbytes_to_read <= block_size_aligned, "[aio] invalid block offset");
      ASSERT_THROW(slice_offset_f + nbytes_to_read <= fsize,              "[aio] invalid block nbytes");

      streamer.request_buffer(freader, data + slice_offset_b, slice_offset_f, nbytes_to_read);
    }
  }

  void submit_all_jobs(int64_t i = -1)
  {
    i = i < 0 ? random_uint64(NUM_BLOCKS) : i;
    for (uint64_t j = 0; j < NUM_CONCURRENT_BLOCKS; ++j) {
      const vec3i block_index = random_grid_index(block_index_space);
      submit_one_job((i + j) % NUM_BLOCKS, block_index);
    }
    streamer.run_for_all();
  }

  void wait_all_jobs()
  {
    streamer.wait_for_all();
  }

  char* access_voxel(const uint64_t bidx, const vec3i& coord) 
  {
    ASSERT_THROW(bidx < NUM_BLOCKS, "[aio] incorrect");
    ASSERT_THROW(blocks.size() == NUM_BLOCKS, "[aio] incorrect");

    const auto& bounds = blocks[bidx].bounds_with_ghost;
    ASSERT_THROW(is_in_bounds(coord, bounds), "[aio] requested voxel is outside the selected block");
    const uint64_t dist = bidx * block_size_aligned + flatten_grid_index(coord - bounds.lower, bounds.size()) * element_size;
    return block_data + dist;
  }

  uint64_t locate_voxel(const uint64_t bidx, const uint64_t voxel_offset)
  {
    return blocks[bidx].offset + voxel_offset;
  }
};

/////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////

#ifdef ENABLE_OPENVKL

static VKLDevice
createOpenVKLDevice()
{
  static VKLDevice device = nullptr;

  if (!device) {
#if openvkl_OLD_API
    vklLoadModule("cpu_device");
    device = vklNewDevice("cpu");
    vklCommitDevice(device);
    vklSetCurrentDevice(device);
#else
    vklLoadModule("cpu_device");
    device = vklNewDevice("cpu");
    vklCommitDevice(device);
#endif
  }

  return device;
}

static VKLDevice device = createOpenVKLDevice();
static std::shared_ptr<openvkl::testing::TestingVolume> testing;

#if SAMPLE_WITH_TRILINEAR_INTERPOLATION
static const VKLFilter vklfilter = VKL_FILTER_TRILINEAR;
#else
static const VKLFilter vklfilter = VKL_FILTER_NEAREST;
#endif

#if openvkl_OLD_API
#define DEVICE_COMMA
#define DEVICE
#else
#define DEVICE_COMMA device,
#define DEVICE device
#endif

OpenVKLSampler::OpenVKLSampler()
{
  using namespace openvkl::testing;

  const float boundingBoxSize = 2.f;
  rkcommon::math::vec3i dimensions(128);
  rkcommon::math::vec3f gridOrigin(0);
  rkcommon::math::vec3f gridSpacing(1.f / rkcommon::math::vec3f(dimensions));
  openvkl::testing::ProceduralStructuredRegularVolume<>::generateGridParameters(dimensions, boundingBoxSize, gridOrigin, gridSpacing);

  // testing = std::make_shared<XYZStructuredRegularVolumeFloat>(DEVICE_COMMA dimensions, gridOrigin, gridSpacing);
  // testing = std::make_shared<SphereStructuredRegularVolumeFloat>(DEVICE_COMMA dimensions, gridOrigin, gridSpacing);
  // testing = std::make_shared<WaveletStructuredRegularVolumeFloat>(DEVICE_COMMA dimensions, gridOrigin, gridSpacing);
  // testing = std::make_shared<XYZUnstructuredProceduralVolume>(DEVICE_COMMA dimensions, gridOrigin, gridSpacing, VKL_HEXAHEDRON);
  // testing = std::make_shared<SphereUnstructuredProceduralVolume>(DEVICE_COMMA dimensions, gridOrigin, gridSpacing, VKL_HEXAHEDRON);
  // testing = std::make_shared<WaveletUnstructuredProceduralVolume>(DEVICE_COMMA dimensions, gridOrigin, gridSpacing, VKL_HEXAHEDRON);
  // testing = std::make_shared<XYZVdbVolumeFloat>(DEVICE_COMMA dimensions, gridOrigin, gridSpacing);
  // testing = std::make_shared<SphereVdbVolumeFloat>(DEVICE_COMMA dimensions, gridOrigin, gridSpacing);
  testing = std::make_shared<WaveletVdbVolumeFloat>(DEVICE_COMMA dimensions, gridOrigin, gridSpacing);

  m_volume = testing->getVKLVolume(DEVICE);
  vklSetFloat((VKLVolume&)m_volume, "background", 0.f);
  vklSetInt((VKLVolume&)m_volume, "filter", vklfilter);
  vklCommit((VKLVolume&)m_volume);

  m_sampler = vklNewSampler((VKLVolume&)m_volume);
  vklCommit((VKLSampler&)m_sampler);

  m_value_range.lower = testing->getComputedValueRange().lower;
  m_value_range.upper = testing->getComputedValueRange().upper;

  m_dims.x = dimensions.x;
  m_dims.y = dimensions.y;
  m_dims.z = dimensions.z;

  m_cell_centered = false;

  const auto bbox = vklGetBoundingBox((VKLVolume&)m_volume);
  m_bbox = (box3f&)bbox;
}

// VDB loader
OpenVKLSampler::OpenVKLSampler(const std::string& filename, const std::string& field)
{
  using namespace openvkl::testing;
  testing = std::shared_ptr<OpenVdbVolume>(OpenVdbVolume::loadVdbFile(DEVICE_COMMA filename, field, VKL_FILTER_TRILINEAR));

  m_volume = testing->getVKLVolume(DEVICE);
  vklSetFloat((VKLVolume&)m_volume, "background", 0.f);
  vklSetInt((VKLVolume&)m_volume, "filter", vklfilter);
  vklCommit((VKLVolume&)m_volume);

  m_sampler = vklNewSampler((VKLVolume&)m_volume);
  vklCommit((VKLSampler&)m_sampler);

  m_value_range.lower = testing->getComputedValueRange().lower;
  m_value_range.upper = testing->getComputedValueRange().upper;

  vkl_box3f bbox = vklGetBoundingBox((VKLVolume&)m_volume);
  m_dims.x = bbox.upper.x - bbox.lower.x;
  m_dims.y = bbox.upper.y - bbox.lower.y;
  m_dims.z = bbox.upper.z - bbox.lower.z;
  m_bbox = (box3f&)bbox;

  m_cell_centered = false;
}

// regular grid
OpenVKLSampler::OpenVKLSampler(const MultiVolume& desc, bool save_volume, bool skip_texture)
{
  m_static_impl = std::make_shared<StaticSampler>(desc, save_volume, skip_texture);
  m_type = m_static_impl->type();
  m_dims = m_static_impl->dims();
  m_tex = m_static_impl->texture();
  create();
}

// downsampled regular grid
OpenVKLSampler::OpenVKLSampler(const MultiVolume& desc, bool save_volume, const vec3i dims)
{
  m_static_impl = std::make_shared<StaticSampler>(desc, save_volume, true);
  m_type = m_static_impl->type();
  m_dims = m_static_impl->dims();
  create();

  m_dims = dims;
  const uint64_t count = (size_t)dims.x * dims.y * dims.z;
  const vec3f rdims = 1.f / vec3f(dims-1.f); // in openvkl, volumes are node centered by default.

  std::unique_ptr<float[]> data(new float[count]);

  const auto value_scale = 1.f / (m_value_range.upper - m_value_range.lower);

  // compute a downsampled volume
  tbb::parallel_for(size_t(0), count, [&](size_t i) {
    const int x = i % dims.x;
    const int y = (i % ((size_t)dims.x * dims.y)) / dims.x;
    const int z = i / ((size_t)dims.x * dims.y);
    const vec3f p = vec3f(x, y, z) * rdims * m_bbox.size() + m_bbox.lower;
    const float v = vklComputeSample((VKLSampler&)m_sampler, (vkl_vec3f*)&p);
    data[i] = (v - m_value_range.lower) * value_scale;
  });

  CreateArray3DScalar<float>(m_downsampled_array, m_downsampled_texture, dims, 
                             SAMPLE_WITH_TRILINEAR_INTERPOLATION, data.get());

  m_tex = m_downsampled_texture;
}

void
OpenVKLSampler::create()
{
  auto& source = *m_static_impl;

  VKLDataType type;
  switch (source.type()) {
  case VALUE_TYPE_UINT8:  type = VKL_UCHAR;  break;
  case VALUE_TYPE_INT8:   type = VKL_CHAR;   break;
  case VALUE_TYPE_UINT16: type = VKL_USHORT; break;
  case VALUE_TYPE_INT16:  type = VKL_SHORT;  break;
  case VALUE_TYPE_UINT32: type = VKL_UINT;   break;
  case VALUE_TYPE_INT32:  type = VKL_INT;    break;
  case VALUE_TYPE_FLOAT:  type = VKL_FLOAT;  break;
  case VALUE_TYPE_DOUBLE: type = VKL_DOUBLE; break;
  default: throw std::runtime_error("unsupported data type");
  }

  m_volume = vklNewVolume(DEVICE_COMMA "structuredRegular");
  vklSetVec3i((VKLVolume&)m_volume, "dimensions", source.dims().x, source.dims().y, source.dims().z);
  vklSetVec3f((VKLVolume&)m_volume, "gridOrigin", 0, 0, 0);
  vklSetVec3f((VKLVolume&)m_volume, "gridSpacing", 1.f, 1.f, 1.f);
  vklSetFloat((VKLVolume&)m_volume, "background", 0.f);
  vklSetInt((VKLVolume&)m_volume, "filter", vklfilter);

  VKLData data = vklNewData(DEVICE_COMMA source.dims().long_product(), type, source.data(/*timestamp=*/0), VKL_DATA_SHARED_BUFFER);
  vklSetData((VKLVolume&)m_volume, "data", data);
  vklRelease(data);

  vklCommit((VKLVolume&)m_volume);

  m_sampler = vklNewSampler((VKLVolume&)m_volume);
  vklCommit((VKLSampler&)m_sampler);

  m_value_range.lower = source.lower();
  m_value_range.upper = source.upper();

  const auto bbox = vklGetBoundingBox((VKLVolume&)m_volume);
  m_bbox = (box3f&)bbox;
}

void
OpenVKLSampler::sample(void* d_coord, void* d_value, size_t batch_size, const vec3f& lower, const vec3f& upper, cudaStream_t stream)
{
  m_coords.resize(batch_size);
  m_values.resize(batch_size);

  const auto value_scale = 1.f / (m_value_range.upper - m_value_range.lower);
  const auto& bbox = m_bbox;
  const auto ncbox = box3f(bbox.lower + 0.5f, bbox.upper + 0.5f);
  const auto ccbox = box3f(bbox.lower, bbox.upper + 1.f);

  random_hbuffer_uniform((float*)d_coord, (float*)m_coords.data(), batch_size * 3);

  tbb::parallel_for(size_t(0), batch_size, [&](size_t i) {
    const vec3f uniform_p = m_coords[i];

    // sample within [0,1)^3
    const vec3f p = uniform_p * (upper - lower) + lower;
    m_coords[i] = p;

    // convert from cell center to node center
    const vec3f ccp = clamp(p * ccbox.size() + ccbox.lower, ncbox.lower, ncbox.upper);
    const vec3f ncp = m_cell_centered ? ccp-0.5f : clamp(p * bbox.size() + bbox.lower, bbox.lower, bbox.upper);

    // sample the volume
    const float v = vklComputeSample((VKLSampler&)m_sampler, (vkl_vec3f*)&ncp);
    m_values[i] = (v - m_value_range.lower) * value_scale;

#if 0 // verify correctness
    const auto accessor = [data=(float*)m_static_impl->data(/*timestamp=*/0), dims=m_dims] (vec3i vidx) {
      return data[vidx.x + vidx.y * dims.x + vidx.z * dims.x * dims.y];
    };
#if SAMPLE_WITH_TRILINEAR_INTERPOLATION
    // verify trilinear interpolation
    const float vr = trilinear_vkl(ccp, m_dims, accessor);
    assert(isclose(vr, v));
#else
    // verify nearest interpolation
    const float vr = nearest_vkl(ccp, m_dims, accessor);
    assert(vr == v);
#endif
#endif
  });

#if 0
  std::vector<float> cuda_value(batch_size);
  m_static_impl->sample_with_inputs(m_coords.data(), d_value, sizeof(float) * batch_size, stream);
  CUDA_CHECK(cudaMemcpyAsync(cuda_value.data(), d_value, sizeof(float) * batch_size, cudaMemcpyDeviceToHost, stream));
  CUDA_SYNC_CHECK();
  for (int i = 0; i < batch_size; ++i) {
    if (!isclose(m_values[i], cuda_value[i])) {
      log << "(" << m_coords[i].x*m_dims.x << "," << m_coords[i].y*m_dims.y << "," << m_coords[i].z*m_dims.z << ") " << m_values[i] << " " << cuda_value[i] << std::endl;
    }
  }
#endif

  CUDA_CHECK(cudaMemcpyAsync(d_coord, m_coords.data(), sizeof(vec3f) * batch_size, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_value, m_values.data(), sizeof(float) * batch_size, cudaMemcpyHostToDevice, stream));
}

void 
OpenVKLSampler::sample_grid(void* d_coords, void* d_values, vec3i grid_origin, vec3i grid_dims, vec3f grid_spacing, cudaStream_t stream)
{
  generate_grid_coords((float*)d_coords, grid_origin, grid_dims, grid_spacing, stream);

  const size_t batch_size = grid_dims.long_product();
  m_coords.resize(batch_size);
  m_values.resize(batch_size);

  CUDA_CHECK(cudaMemcpyAsync(m_coords.data(), d_coords, sizeof(vec3f) * batch_size, cudaMemcpyDeviceToHost, stream));

  sample_with_inputs(m_coords.data(), m_values.data(), batch_size, stream);

  CUDA_CHECK(cudaMemcpyAsync(d_values, m_values.data(), sizeof(float) * batch_size, cudaMemcpyHostToDevice, stream));
}

void
OpenVKLSampler::sample_with_inputs(const vec3f* h_coords, float* h_values, size_t batch_size, cudaStream_t stream)
{
  const auto value_scale = 1.f / (m_value_range.upper - m_value_range.lower);
  const auto& bbox = m_bbox;
  const auto ncbox = box3f(bbox.lower + 0.5f, bbox.upper + 0.5f);
  const auto ccbox = box3f(bbox.lower, bbox.upper + 1.f);

  tbb::parallel_for(size_t(0), batch_size, [&](size_t i) {

    const vec3f p = m_cell_centered 
      ? clamp(h_coords[i] * ccbox.size() + ccbox.lower, ncbox.lower, ncbox.upper)-0.5f // convert from cell center to node center
      : clamp(h_coords[i] * bbox.size()  + bbox.lower,  bbox.lower,  bbox.upper );     // or directly sampling as node centered volume

    const float v = vklComputeSample((VKLSampler&)m_sampler, (vkl_vec3f*)&p);
    h_values[i] = (v - m_value_range.lower) * value_scale;

  });
}

#endif // ENABLE_OPENVKL

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

void
sample_streaming_grid(void* d_coords,
                      void* d_values,
                      vec3i grid_origin,
                      vec3i grid_dims,
                      vec3f grid_spacing,
                      cudaStream_t stream,
                      std::vector<float>& m_values,
                      vidi::FileMap& m_reader,
                      const size_t m_offset,
                      const ValueType m_type,
                      const vec3i m_dims,
                      const range1f m_value_range)
{
  constexpr bool trilinear = false;

  // include ghost regions
  if constexpr (trilinear)
  {
    vec3i lower = max(grid_origin-1,           vec3i(0) );
    vec3i upper = max(grid_origin+grid_dims+1, grid_dims);
    grid_origin = lower;
    grid_dims = upper - lower;
  }

  // ASSUMPTION: the CPU can hold the entire block !!!
  const size_t batch_size = grid_dims.long_product();
  m_values.resize(batch_size);

  // step 1. copy the entire grid to CPU
  const auto elem_size = value_type_size(m_type);
  std::vector<char> rawdata(elem_size * batch_size);
  size_t pos = 0;
  for (int z = 0; z < grid_dims.z; ++z) {
    for (int y = 0; y < grid_dims.y; ++y) {
      const auto offset = flatten_grid_index(vec3i(0, y, z) + grid_origin, m_dims);
      filemap_random_read(m_reader, elem_size * offset + m_offset, 
                                    elem_size * pos + rawdata.data(), 
                                    elem_size * grid_dims.x);
      pos += grid_dims.x;
    }
  }

  // step 2. sampling
  const auto scale = 1.f / m_value_range.span();
  const auto accessor = [&] (const vec3i& ip) {
    ASSERT_THROW(is_in_grid(ip - grid_origin, grid_dims),  "[virtual memory] invalid sample coordinate");
    const auto offset = flatten_grid_index(ip - grid_origin, grid_dims);
    const float v = read_typed_pointer(rawdata.data(), offset, m_type);
    // NOTE: do data normalization before interpolation can significantly reduce error !!!! 
    return clamp((v - m_value_range.lower) * scale, 0.f, 1.f); // normalize
  };

  const auto fdims = vec3f(m_dims);
  tbb::parallel_for(size_t(0), batch_size, [&](size_t i) {
    const uint64_t stride = (uint64_t)grid_dims.x * grid_dims.y;
    const int32_t x = grid_origin.x +  i % grid_dims.x;
    const int32_t y = grid_origin.y + (i % stride) / grid_dims.x;
    const int32_t z = grid_origin.z +  i / stride;
    const vec3f fp = (vec3f(x,y,z) + 0.5f) * grid_spacing; // sample within [0,1)^3 
    const vec3f ccp = clamp(fp * fdims, vec3f(0.5f), fdims-0.5f);
    m_values[i] = trilinear ? trilinear_vkl(ccp, m_dims, accessor) : nearest_vkl(ccp, m_dims, accessor);
  });

  // step 3. generate coordinates
  generate_grid_coords((float*)d_coords, grid_origin, grid_dims, grid_spacing, stream);

  // step 4. copy values to gpu
  CUDA_CHECK(cudaMemcpyAsync(d_values, m_values.data(), sizeof(float) * batch_size, cudaMemcpyHostToDevice, stream));
}

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

#ifdef ENABLE_OUT_OF_CORE

OutOfCoreSampler::OutOfCoreSampler(const MultiVolume& desc)
{
  m_dims = desc.dims;
  m_type = desc.type;
  m_value_range = desc.range;

  const auto& file = desc.data[0];
  assert(file.bigendian == false && "only support small endian");
  m_offset = file.offset;
  m_reader = vidi::filemap_read_create_async(file.filename);

  int VNR_NUM_CONCURRENT_BLOCKS = 1024;
  if (const char* env_p = std::getenv("VNR_NUM_CONCURRENT_BLOCKS")) {
    VNR_NUM_CONCURRENT_BLOCKS = std::stoi(env_p);
  }
  int VNR_NUM_BLOCKS = VNR_NUM_CONCURRENT_BLOCKS*64;
  if (const char* env_p = std::getenv("VNR_NUM_BLOCKS")) {
    VNR_NUM_BLOCKS = std::stoi(env_p);
  }
  m_randbuf = std::make_shared<RandomBuffer>(m_reader, desc.dims, value_type_size(desc.type), VNR_NUM_CONCURRENT_BLOCKS, VNR_NUM_BLOCKS);
}

void
OutOfCoreSampler::sample(void* d_coord, void* d_value, size_t batch_size, const vec3f& lower, const vec3f& upper, cudaStream_t stream)
{
  if (m_value_range.is_empty()) {
    throw std::runtime_error("a valid value range must be provided");
  }

  const vec3f rfdims = 1.f / vec3f(m_dims);
  const float vscale = 1.f / m_value_range.span();
  auto& randbuf = *m_randbuf;

  m_coords.resize(batch_size);
  m_values.resize(batch_size);
  m_random_bidx.resize(batch_size);
  m_random_vidx.resize(batch_size);

  random_hbuffer_uniform((float*)m_coords.data(), batch_size * 3);
  random_hbuffer_uniform(m_random_bidx.data(), batch_size);
  random_hbuffer_uniform(m_random_vidx.data(), batch_size);

  randbuf.wait_all_jobs();

  tbb::parallel_for(size_t(0), batch_size, [&](size_t i) {
    const uint64_t bidx = m_random_bidx[i] * randbuf.NUM_BLOCKS;
    ASSERT_THROW(bidx < randbuf.NUM_BLOCKS, "[aio] invalid block index");

    const uint64_t vidx = m_random_vidx[i] * randbuf.blocks[bidx].length;
    ASSERT_THROW(vidx < randbuf.blocks[bidx].length, "[aio] invalid voxel index");

    // find the point index
    const vec3i voxel = to_grid_index(randbuf.locate_voxel(bidx, vidx), m_dims);  // voxel index in file

    // randomly sample a point within the cell
    const vec3f p = m_coords[i] + vec3f(voxel);
    m_coords[i] = p * rfdims * (upper - lower) + lower; // normalize to [0,1)

    // trilinear
    const auto accessor = [&] (const vec3i& idx) {
      ASSERT_THROW(is_in_bounds(idx, randbuf.blocks[bidx].bounds_with_ghost), "[aio] incorrect");
      const float v = read_typed_pointer(randbuf.access_voxel(bidx, idx), 0, m_type);
      return clamp((v - m_value_range.lower) * vscale, 0.f, 1.f); // normalize before interpolation
    };

    const vec3f ccp = clamp(p, vec3f(0.5f), vec3f(m_dims)-0.5f);
#if SAMPLE_WITH_TRILINEAR_INTERPOLATION
    m_values[i] = trilinear_vkl(ccp, m_dims, accessor); 
#else
    m_values[i] = nearest_vkl  (ccp, m_dims, accessor); 
#endif
  });

  randbuf.submit_all_jobs();

  CUDA_CHECK(cudaMemcpyAsync(d_coord, m_coords.data(), sizeof(vec3f) * batch_size, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_value, m_values.data(), sizeof(float) * batch_size, cudaMemcpyHostToDevice, stream));
}

void 
OutOfCoreSampler::sample_grid(void* d_coords, void* d_values, vec3i grid_origin, vec3i grid_dims, vec3f grid_spacing, cudaStream_t stream)
{
  // ASSUMPTION: the CPU can hold the entire block !!!
  sample_streaming_grid(d_coords, d_values, grid_origin, grid_dims, grid_spacing, stream, m_values, m_reader, m_offset, m_type, m_dims, m_value_range);
}

#endif // ENABLE_OUT_OF_CORE

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

#ifdef ENABLE_OUT_OF_CORE

VirtualMemorySampler::VirtualMemorySampler(const MultiVolume& desc)
{
  m_dims = desc.dims;
  m_type = desc.type;
  m_value_range = desc.range;

  const auto& file = desc.data[0];
  assert(file.bigendian == false && "only support small endian");
  m_offset = file.offset;
  m_reader = vidi::filemap_read_create(file.filename);

  m_fdims = vec3f(m_dims);
  m_value_scale = 1.f / (m_value_range.upper - m_value_range.lower);
  m_elem_size = value_type_size(m_type);
}

void
VirtualMemorySampler::sample(void* d_coord, void* d_value, size_t batch_size, const vec3f& lower, const vec3f& upper, cudaStream_t stream)
{
  if (m_value_range.is_empty())
    throw std::runtime_error("a valid value range must be provided");

  m_coords.resize(batch_size);
  m_values.resize(batch_size);

  const auto accessor = [&] (const vec3i& ip) {
    char elem_data[8];
    const auto elem_index = (size_t)ip.x + (size_t)ip.y * (size_t)m_dims.x + (size_t)ip.z * (size_t)m_dims.y * (size_t)m_dims.x;
    filemap_random_read(m_reader, m_offset + elem_index * m_elem_size, elem_data, m_elem_size);
    // do data normalization before interpolation can significantly reduce error !!!! 
    const float v = read_typed_pointer(elem_data, 0, m_type);
    return clamp((v - m_value_range.lower) * m_value_scale, 0.f, 1.f);
  };

  random_hbuffer_uniform((float*)d_coord, (float*)m_coords.data(), batch_size * 3);

  tbb::parallel_for(size_t(0), batch_size, [&](size_t i) {
    const vec3f uniform_p = m_coords[i];

    // sample within [0,1)^3 
    const vec3f fp = uniform_p * (upper - lower) + lower;
    m_coords[i] = fp; // simulate nearest interpolation

    // interpolation: TODO currently doing I/O blocking
    const vec3f ccp = clamp(fp * m_fdims, vec3f(0.5f), m_fdims-0.5f);
#if SAMPLE_WITH_TRILINEAR_INTERPOLATION
    m_values[i] = trilinear_vkl(ccp, m_dims, accessor);
#else
    m_values[i] = nearest_vkl  (ccp, m_dims, accessor); 
#endif
  });

  CUDA_CHECK(cudaMemcpyAsync(d_coord, m_coords.data(), sizeof(vec3f) * batch_size, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_value, m_values.data(), sizeof(float) * batch_size, cudaMemcpyHostToDevice, stream));
}

void 
VirtualMemorySampler::sample_grid(void* d_coords, void* d_values, vec3i grid_origin, vec3i grid_dims, vec3f grid_spacing, cudaStream_t stream)
{
  // ASSUMPTION: the CPU can hold the entire block !!!
  sample_streaming_grid(d_coords, d_values, grid_origin, grid_dims, grid_spacing, stream, m_values, m_reader, m_offset, m_type, m_dims, m_value_range);
}

#endif // ENABLE_OUT_OF_CORE

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

void
Sampler::load(const MultiVolume& desc, std::string training_mode, bool save_volume)
{
  /* GPU-based */
  if (training_mode == "GPU") {
    impl = std::make_shared<StaticSampler>(desc, save_volume, false);
    m_dims = impl->dims();
  }

#ifdef ENABLE_OUT_OF_CORE

  /* CPU-based, virtual memory, no ground truth */
  else if (training_mode == "VIRTUAL_MEMORY") {
    impl = std::make_shared<VirtualMemorySampler>(desc);
    m_dims = gdt::min(vec3i(1024), vec3i(desc.dims));
  }

  /* out-of-core-steaming */
  else if (training_mode == "OUT_OF_CORE") {
    impl = std::make_shared<OutOfCoreSampler>(desc);
    m_dims = gdt::min(vec3i(1024), vec3i(desc.dims));
  }

#endif // ENABLE_OUT_OF_CORE

#ifdef ENABLE_OPENVKL

  /* CPU-based, openvkl, no ground truth */
  else if (training_mode == "OPENVKL") {
    impl = std::make_shared<OpenVKLSampler>(desc, save_volume, true);
    m_dims = gdt::min(vec3i(1024), vec3i(desc.dims));
  }

  /* use OpenVKL sampling but with a ground truth at the original resolution */
  else if (training_mode == "OPENVKL_GT_ORIGINAL_RESOLUTION") {
    impl = std::make_shared<OpenVKLSampler>(desc, /*save_volume=*/false, false);
    m_dims = impl->dims();
  }

  /* use OpenVKL sampling but with a ground truth at the original resolution */
  else if (training_mode == "OPENVKL_GT_DOWNSAMPLE_RESOLUTION") {
    m_dims = vec3i(desc.dims) / 8; // downsampled by 8x
    impl = std::make_shared<OpenVKLSampler>(desc, /*save_volume=*/false, m_dims);
  }

  /* using OpenVKL to support irregular datasets */
  else if (training_mode == "OPENVKL_IRREGULAR") {
    impl = std::make_shared<OpenVKLSampler>();
    // impl = std::make_shared<OpenVKLSampler>("/home/qwu/Work/datasets/openvdb/bunny_cloud.vdb", "density");
    // impl = std::make_shared<OpenVKLSampler>("/home/qwu/Work/datasets/openvdb/wdas_cloud.vdb", "density");
    m_dims = vec3i(impl->dims()) * 1024 / gdt::reduce_max(vec3i(impl->dims()));
    m_transform = affine3f::translate(-vec3f(m_dims) / 2.f) * affine3f::scale(vec3f(m_dims));
    return;
  }

#endif // ENABLE_OPENVKL

  else if (training_mode == "NOTHING") {
    impl = std::make_shared<StaticSampler>(desc.dims, desc.type);
    m_dims = impl->dims();
  }

  else throw std::runtime_error("unknown mode");

  m_transform = affine3f::translate(vec3f(desc.dims) * -0.5f) * affine3f::scale(vec3f(desc.dims));
}

} // namespace vnr
