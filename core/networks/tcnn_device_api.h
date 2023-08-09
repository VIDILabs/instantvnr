//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //
#pragma once
#ifndef TCNN_DEVICE_API_H
#define TCNN_DEVICE_API_H

#ifndef ENABLE_IN_SHADER
#error "<tcnn_device_api.h> works only if in-shader inference has been enabled"
#endif

#include "tcnn_network.h"

#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/networks/fully_fused_mlp.h>

#include <memory>

/* namespace instant neural volume */
namespace vnr {
namespace tcnn_impl {

using TCNN_NAMESPACE :: FullyFusedMLP;
using TCNN_NAMESPACE :: GridEncodingTemplated;
using TCNN_NAMESPACE :: Activation;
using TCNN_NAMESPACE :: InterpolationType;
using TCNN_NAMESPACE :: PitchedPtr;
using TCNN_NAMESPACE :: GridType;
using TCNN_NAMESPACE :: GridOffsetTable;
using TCNN_NAMESPACE :: HashType;
using TCNN_NAMESPACE :: MatrixLayout;
using TCNN_NAMESPACE :: RM;
using TCNN_NAMESPACE :: CM;

template<typename Type>
using Matrix = GPUMatrixDynamic<Type>;

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

// NOTE: EncoderCtx is expensive to copy, use references always.

template<typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, HashType HASH_TYPE>
struct EncoderCtx {
  uint32_t num_levels;
  uint32_t num_grid_features;
  /*uint32_t *hashmap_offset_table;*/
  GridOffsetTable offset_table;
  uint32_t base_resolution;
  float log2_per_level_scale;
  float quantize_threshold;
  float max_level;
  InterpolationType interpolation_type;
  GridType grid_type;
  T* __restrict__ grid;
};

// NOTE: output has to be zero-initialized
template<typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, HashType HASH_TYPE>
__device__ void encode(
  const EncoderCtx<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, HASH_TYPE>& ctx, 
  const uint32_t level, 
  const float* __restrict__ input, 
  T* __restrict__ output
);

}
}

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

namespace vnr {
namespace tcnn_impl {

static const char* c_str(const std::string& s) { return s.c_str(); }
template<typename T> static T c_str(T s) { return s; }

#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security" 
#endif
template<typename... Ts>
static std::string stringf(const std::string& format, Ts... rest) {
  int64_t sz = snprintf(NULL, 0, format.c_str(), c_str(rest)...);
  char* bf = static_cast<char*>(malloc(sz + 1));
  snprintf(bf, sz + 1, format.c_str(), c_str(rest)...);
  std::string ret(bf);
  free(bf);
  return ret;
}
#ifndef _WIN32
#pragma GCC diagnostic pop
#endif

template<typename T, int WIDTH>
struct NetworkCtx {
  Activation activation;
  Activation output_activation;
  T* __restrict__ weights;
  uint32_t n_hidden_matmuls;
  uint32_t n_input_width;
};

}
}

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

namespace vnr {
namespace tcnn_impl {

template<typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t WIDTH, HashType HASH_TYPE = HashType::CoherentPrime>
struct DeviceNeuralVolume {
private:
  using EncoderType = GridEncodingTemplated<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, HASH_TYPE>;
  using NetworkType = FullyFusedMLP<T, WIDTH>;

  static_assert(WIDTH < 256, "maximum WIDTH == 128");
  static_assert(WIDTH % 16 == 0, "Width must be a multiply of 16.");
  static_assert(N_POS_DIMS == 3, "N_POS_DIMS must be 3");

  constexpr static uint32_t SKEW = (WIDTH % 16 == 0) ? 8 : 0; // <- always going to be 8
  constexpr static uint32_t INPUT_SKEW = 8;                   // <- likewise with inputs
  constexpr static uint32_t N_BLOCK_ROWS = WIDTH / 16;
  constexpr static uint32_t N_ITERS = 2 * N_BLOCK_ROWS;

private:
  mutable EncoderCtx<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, HASH_TYPE> enc;
  mutable NetworkCtx<T, WIDTH> mlp;
  NetworkWithInputEncoding* handler{ nullptr };

public:
  void create_encoder_ctx(/*GPUMemory<uint32_t>& offset_table_device*/) const;
  void create_network_ctx() const;

  DeviceNeuralVolume(void* h);

  template<typename V, typename K, typename... Types>
  void launch_general(const V& This, K kernel, cudaStream_t stream, uint32_t requested_batch_size, Types... args) const;

  template<typename K, typename... Types> void launch1D(K kernel, cudaStream_t stream, int32_t width, Types... args) const;
  template<typename K, typename... Types> void launch2D(K kernel, cudaStream_t stream, int32_t width, int32_t height, Types... args) const;
  template<typename K, typename... Types> void launch3D(K kernel, cudaStream_t stream, int32_t width, int32_t height, int32_t depth, Types... args) const;

  __device__ __forceinline__ void init() const {}

  __device__ T sample(float3 coordinate) const;
};

template<typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t WIDTH, HashType HASH_TYPE>
void DeviceNeuralVolume<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, WIDTH, HASH_TYPE>::create_encoder_ctx(/*GPUMemory<uint32_t>& offset_table_device*/) const {
  auto* encoder = dynamic_cast<EncoderType*>(handler->m_encoding.get());
  ASSERT_THROW(encoder, "wrong encoding type");
  
  // // Uploade to GPU only once
	// offset_table_device.resize(encoder->m_n_levels + 1);
	// CUDA_CHECK(cudaMemcpy(offset_table_device.data(), encoder->m_offset_table.data, (encoder->m_n_levels+1) * sizeof(uint32_t), cudaMemcpyHostToDevice));

  enc.num_levels = encoder->m_n_levels;
  enc.num_grid_features = encoder->m_n_features;
  /*enc.hashmap_offset_table = offset_table_device.data();*/
  enc.offset_table = encoder->m_offset_table;
  enc.base_resolution = encoder->m_base_resolution;
  enc.log2_per_level_scale = std::log2(encoder->m_per_level_scale);
  enc.quantize_threshold = encoder->m_quantize_threshold;
  enc.max_level = encoder->m_max_level;
  enc.interpolation_type = encoder->m_interpolation_type;
  enc.grid_type = encoder->m_grid_type;
  enc.grid = encoder->inference_params();
}

template<typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t WIDTH, HashType HASH_TYPE>
void DeviceNeuralVolume<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, WIDTH, HASH_TYPE>::create_network_ctx() const {
  auto* network = dynamic_cast<NetworkType*>(handler->m_network.get());
  ASSERT_THROW(network, "wrong network type");

  mlp.activation = network->m_activation;
  mlp.output_activation = network->m_output_activation;
  mlp.weights = network->input_weight_matrix(true).data();
  mlp.n_hidden_matmuls = network->m_n_hidden_matmuls;
  mlp.n_input_width = network->m_input_width;
}

template<typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t WIDTH, HashType HASH_TYPE>
DeviceNeuralVolume<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, WIDTH, HASH_TYPE>::DeviceNeuralVolume(void* h) : handler((NetworkWithInputEncoding*)h) {
  auto* h_enc = dynamic_cast<EncoderType*>(handler->m_encoding.get());
  ASSERT_THROW(h_enc, "wrong encoding type");
  auto* h_mlp = dynamic_cast<NetworkType*>(handler->m_network.get());
  ASSERT_THROW(h_mlp, "wrong network type");

  // Validate ENC
  ASSERT_THROW(h_enc->m_max_level_gpu == nullptr, "null pointer 'm_max_level_gpu' expected");
  
  // Validate MLP
  static_assert(WIDTH % 16 == 0, "Width must be a multiply of 16.");
  ASSERT_THROW(h_mlp->m_input_width % 16 == 0, "Inputs must have a multiple-of-16 elements.");
  const auto& w = h_mlp->input_weight_matrix(true);
  ASSERT_THROW(w.rows() == WIDTH, "The fully fused forward pass only works with WIDTH-sized matrices.");
  ASSERT_THROW(w.cols() % 16 == 0, stringf("weights must have a multiple-of-16 number of columns, receiving %d columns.", w.cols()).c_str());
  switch (h_mlp->m_activation) {
    case Activation::None:        break;
    case Activation::Exponential: break;
    case Activation::Sigmoid:     break;
    case Activation::ReLU:        break;
    case Activation::Squareplus:  break;
    case Activation::Softplus:    break;
    default: throw std::runtime_error{"Unsupported activation."};
  }
  
  // Validate Both 
  ASSERT_THROW(h_enc->padded_output_width() == h_mlp->m_input_width, "encoder and network have different width");
  ASSERT_THROW(h_enc->padded_output_width() != 0, "incorrect output dimension");
}

static void check_shmem(cudaError_t error) {
	if (error != cudaSuccess) {
		throw std::runtime_error{"DeviceNeuralVolume: insufficient shared memory available on the GPU. Reduce `n_neurons` instead."};
	}
}

template<typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t WIDTH, HashType HASH_TYPE>
template<typename V, typename K, typename... Types>
void DeviceNeuralVolume<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, WIDTH, HASH_TYPE>::launch_general(const V& This, K kernel, cudaStream_t stream, const uint32_t requested_batch_size, Types... args) const {
  using namespace TCNN_NAMESPACE;

	// GPUMemory<uint32_t> offset_table_device;

  This.create_encoder_ctx(/*offset_table_device*/);
  This.create_network_ctx();

  /* calculate launch dimensions */
  const uint32_t in_width = mlp.n_input_width;
  const uint32_t batch_size = next_multiple(requested_batch_size, (16 * N_ITERS)); // = number of pixels
  ASSERT_THROW(batch_size % (16 * N_ITERS) == 0, stringf("Batch size must be a multiple of %d.", 16 * N_ITERS).c_str());

  const dim3 threads = { 32u, N_BLOCK_ROWS, 1 }; // 32 threads = 1 warp, N_BLOCK_ROWS warps per block for 16 rows, up to 2x 8 warps can share input (does not help vs. 1)
  const uint32_t n_elems_per_block = 16 * N_ITERS;
  const uint32_t n_blocks = div_round_up(batch_size, n_elems_per_block);

  /* calculate shared memory size */
  constexpr uint32_t shmem_size_coord  = sizeof(float ) * (16 * N_ITERS) * N_POS_DIMS;
  constexpr uint32_t shmem_size_output = sizeof(__half) * (16 * N_ITERS) * 16;

  // 16*WIDTH rows of weights (for the last layer; others are in registers only) + 16*WIDTH*BLOCK_DIM_Z*N_ITERS rows of intermediate activations
  size_t shmem_size = sizeof(__half) * (16 + 16 * N_ITERS) * (WIDTH + SKEW); 
  // If the input width is dynamic, the input weight matrix as well as part of the input will live in extra shared memory
  if (in_width != WIDTH) {
    shmem_size = std::max(shmem_size, sizeof(__half) * (WIDTH + 16) * (in_width + INPUT_SKEW));
  }
  shmem_size += shmem_size_coord + shmem_size_output;

  /* launch kernel */
  const dim3 blocks = { n_blocks, 1u, 1u };
  check_shmem(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_size));

  TRACE_CUDA;
  kernel<<<blocks, threads, shmem_size, stream>>>(This, args...);
  TRACE_CUDA;
}

template<typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t WIDTH, HashType HASH_TYPE>
template<typename K, typename... Types>
void DeviceNeuralVolume<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, WIDTH, HASH_TYPE>::launch1D(K kernel, cudaStream_t stream, int32_t width, Types... args) const {
  if (width <= 0) { return; }
  launch_general(*this, kernel, stream, (uint32_t)width, width, args...);
}

template<typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t WIDTH, HashType HASH_TYPE>
template<typename K, typename... Types>
void DeviceNeuralVolume<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, WIDTH, HASH_TYPE>::launch2D(K kernel, cudaStream_t stream, int32_t width, int32_t height, Types... args) const {
  if (width <= 0 || height <= 0) { return; }
  launch_general(*this, kernel, stream, (uint32_t)width*height, width, height, args...);
}

template<typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t WIDTH, HashType HASH_TYPE>
template<typename K, typename... Types>
void DeviceNeuralVolume<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, WIDTH, HASH_TYPE>::launch3D(K kernel, cudaStream_t stream, int32_t width, int32_t height, int32_t depth, Types... args) const {
  if (width <= 0 || height <= 0 || depth <= 0) { return; }
  launch_general(*this, kernel, stream, (uint32_t)width*height*depth, width, height, depth, args...);
}

}
}

namespace vnr {

template<int WIDTH, int N_FEATURES_PER_LEVEL>
using TcnnDeviceVolume = tcnn_impl::DeviceNeuralVolume<tcnn_impl::precision_t, /*N_POS_DIMS=*/3, N_FEATURES_PER_LEVEL, WIDTH>;

} // namespace vnr

#endif // TCNN_DEVICE_API_H
