//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //
#pragma once

#include "tcnn_network.h"

#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/networks/fully_fused_mlp.h>

using TCNN_NAMESPACE :: FullyFusedMLP;
using TCNN_NAMESPACE :: GridEncodingTemplated;
using TCNN_NAMESPACE :: Activation;
using TCNN_NAMESPACE :: InterpolationType;
using TCNN_NAMESPACE :: PitchedPtr;
using TCNN_NAMESPACE :: InterpolationType;
using TCNN_NAMESPACE :: GridType;
using TCNN_NAMESPACE :: WeightUsage;
using TCNN_NAMESPACE :: MatrixLayout;
using TCNN_NAMESPACE :: RM;
using TCNN_NAMESPACE :: CM;

TCNN_NAMESPACE_BEGIN
void check_shmem_error(cudaError_t error);
TCNN_NAMESPACE_END

#include <memory>

/* namespace instant neural volume */
namespace vnr {
namespace tcnn_impl {

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

template<typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL>
struct DeviceNeuralEncoder
{
  using EncoderType = GridEncodingTemplated<T, N_POS_DIMS, N_FEATURES_PER_LEVEL>;
  using PerLevelVec = typename TCNN_NAMESPACE :: vector_t<T, N_FEATURES_PER_LEVEL>;

  DeviceNeuralEncoder(EncoderType* encoder)
    : num_levels(encoder->m_n_levels)
    , num_grid_features(encoder->m_n_features)
    , hashmap_offset_table(encoder->m_hashmap_offsets_table.data())
    , base_resolution(encoder->m_base_resolution)
    , log2_per_level_scale(std::log2(encoder->m_per_level_scale))
    , quantize_threshold(encoder->m_quantize_threshold)
    , max_level(encoder->m_max_level)
    , interpolation_type(encoder->m_interpolation_type)
    , grid_type(encoder->m_grid_type)
    , grid_data(encoder->m_grid_inference)
    , output_layout(encoder->m_output_layout)
    , self(encoder)
  {
    assert(encoder->m_max_level_gpu == nullptr && "null pointer 'm_max_level_gpu' expected");
  }

  __device__ void encode_one_level(const uint32_t level /* the same for all threads */,
                                   const float* __restrict__ input /* float[N_POS_DIMS] */,
                                   T* __restrict__ output_per_level /* T[N_FEATURES_PER_LEVEL] */) const;

  void batch_encode(cudaStream_t stream,
                    uint32_t num_elements,
                    PitchedPtr<const float> inputs,
                    PitchedPtr<T> outputs);

public:
  const uint32_t num_levels;
  const uint32_t num_grid_features;
  const uint32_t* hashmap_offset_table;
  const uint32_t base_resolution;
  const float log2_per_level_scale;
  const float quantize_threshold;
  const float max_level;
  const InterpolationType interpolation_type;
  const GridType grid_type;
  const T* __restrict__ grid_data;

  const MatrixLayout output_layout;

  EncoderType* self;
};

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

template<typename T, int WIDTH>
struct DeviceNeuralNetwork
{
private:
  template<typename Type>
  using Matrix = GPUMatrixDynamic<Type>;

public:
  using NetworkType = FullyFusedMLP<T, WIDTH>;

  DeviceNeuralNetwork(NetworkType* network)
    : activation(network->m_activation)
    , output_activation(network->m_output_activation)
    , weights(network->input_weight_matrix(WeightUsage::Inference).data())
    , n_hidden_matmuls(network->m_n_hidden_matmuls)
    , n_input_width(network->m_input_width)
    , self(network)
  {
    const auto& w = network->input_weight_matrix(WeightUsage::Inference);
    // clang-format off
    if (w.rows() != WIDTH)  throw std::runtime_error{"The fully fused forward pass only works with WIDTH-sized matrices."};
    if (w.cols() % 16 != 0) throw std::runtime_error{std::string("weights must have a multiple-of-16 number of columns. ") + std::to_string(w.cols())};
    // clang-format on
  }

  void batch_inference(cudaStream_t stream, const Matrix<T>& input, Matrix<float>& output);

private:
  void batch_inference_internal(cudaStream_t stream,
                                Activation activation,
                                Activation output_activation,
                                const GPUMatrix<T, RM>& weights,
                                const uint32_t n_hidden_layers,
                                const Matrix<T>& input,
                                Matrix<T>& output);

public:
  const Activation activation;
  const Activation output_activation;
  const T* __restrict__ weights;
  const uint32_t n_hidden_matmuls;
  const uint32_t n_input_width;

  NetworkType* self;
};

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

template<typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t WIDTH>
struct DeviceNeuralVolume
{
  constexpr static uint32_t width() { return WIDTH; }
  constexpr static uint32_t n_pos_dims() { return N_POS_DIMS; }
  constexpr static uint32_t n_features_per_level() { return N_FEATURES_PER_LEVEL; }
  constexpr static uint32_t n_block_dim_z() { return BLOCK_DIM_Z; }
  constexpr static uint32_t n_iters() { return N_ITERS; }

  using EncoderType = GridEncodingTemplated<T, N_POS_DIMS, N_FEATURES_PER_LEVEL>;
  using NetworkType = FullyFusedMLP<T, WIDTH>;

  using EncoderDeviceType = DeviceNeuralEncoder<T, N_POS_DIMS, N_FEATURES_PER_LEVEL>;
  using NetworkDeviceType = DeviceNeuralNetwork<T, WIDTH>;

private:
  template<typename Type>
  using Matrix = GPUMatrixDynamic<Type>;

  static_assert(WIDTH < 256, "maximum WIDTH == 128");
  static_assert(WIDTH % 16 == 0, "Width must be a multiply of 16.");
  static_assert(N_POS_DIMS == 3, "N_POS_DIMS must be 3");

  constexpr static uint32_t SKEW = (WIDTH % 16 == 0) ? 8 : 0; // <- always going to be 8
  constexpr static uint32_t INPUT_SKEW = 8;                   // <- likewise with inputs
  constexpr static uint32_t N_BLOCK_ROWS = WIDTH / 16;
  constexpr static uint32_t N_ITERS = 2 * N_BLOCK_ROWS;
  constexpr static uint32_t BLOCK_DIM_Z = 1;

  struct Validation
  {
    NetworkWithInputEncoding* handler{nullptr};
    Validation() = default;
    Validation(NetworkWithInputEncoding* h)
    {
      handler = h;
      auto enc = dynamic_cast<EncoderType*>(handler->m_encoding.get());
      assert(enc && "wrong encoding type");
      auto mlp = dynamic_cast<NetworkType*>(handler->m_network.get());
      assert(mlp && "wrong network type");
    }
    EncoderType* encoder_ptr() { return dynamic_cast<EncoderType*>(handler->m_encoding.get()); }
    NetworkType* network_ptr() { return dynamic_cast<NetworkType*>(handler->m_network.get()); }
  } validation;

  DeviceNeuralEncoder<T, N_POS_DIMS, N_FEATURES_PER_LEVEL> m_encoder;
  DeviceNeuralNetwork<T, WIDTH> m_network;
  const uint32_t m_n_internal_features;

  void check() const
  {
    if (m_encoder.self->num_encoded_dims() != m_network.self->m_input_width)
      throw std::runtime_error(std::string("encoder and network have different width"));
    assert(m_encoder.self->m_n_padded_output_dims != 0 && "incorrect output dimension");
  }

public:
  DeviceNeuralVolume(EncoderType* h_encoder, NetworkType* h_network)
    : m_encoder(h_encoder), m_network(h_network), m_n_internal_features(h_network->m_input_width)
  {
    check();
  }

  DeviceNeuralVolume(void* h)
    : validation((NetworkWithInputEncoding*)h)
    , m_encoder(validation.encoder_ptr())
    , m_network(validation.network_ptr())
    , m_n_internal_features(validation.network_ptr()->m_input_width)
  {
    check();
  }

  template<typename V, typename K, typename... Types>
  void launch_general(const V& This, K kernel, cudaStream_t stream, uint32_t requested_batch_size, Types... args) const;

  template<typename K, typename... Types> void launch1D(K kernel, cudaStream_t stream, int32_t width, Types... args) const;
  template<typename K, typename... Types> void launch2D(K kernel, cudaStream_t stream, int32_t width, int32_t height, Types... args) const;
  template<typename K, typename... Types> void launch3D(K kernel, cudaStream_t stream, int32_t width, int32_t height, int32_t depth, Types... args) const;

  __device__ __host__ const EncoderDeviceType& encoder() const { return m_encoder; }
  __device__ __host__ const NetworkDeviceType& network() const { return m_network; }

  __device__ __forceinline__ void init() const {}

  __device__ T sample(float3 coordinate) const;

  void batch_sample(cudaStream_t stream, const Matrix<float>& coord, Matrix<float>& output);

private:
  void batch_sample_internal(cudaStream_t stream,
                             const Activation ACTIVATION,
                             const Matrix<float>& coord,
                             Matrix<T>& output);
};

template<typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t WIDTH>
template<typename V, typename K, typename... Types>
void
DeviceNeuralVolume<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, WIDTH>::
launch_general(const V& This, K kernel, cudaStream_t stream, const uint32_t requested_batch_size, Types... args) const
{
  // clang-format off
  using namespace TCNN_NAMESPACE;

  // auto& enc = this->m_encoder;
  auto& mlp = this->m_network;

  /* calculate launch dimensions */
  const uint32_t batch_size = next_multiple(requested_batch_size, (16 * N_ITERS * BLOCK_DIM_Z)); // = number of pixels
  // const uint32_t out_width  = 1;
  const uint32_t in_width   = mlp.n_input_width;

  static_assert(WIDTH % 16 == 0, "Width must be a multiply of 16.");
  if (in_width % 16 != 0) throw std::runtime_error{"Inputs must have a multiple-of-16 elements."};
  if (batch_size % (16 * N_ITERS * BLOCK_DIM_Z) != 0) throw std::runtime_error{"Batch size must be a multiple of " + std::to_string(16 * N_ITERS * BLOCK_DIM_Z) + "."};

  const dim3 threads = { 32u, N_BLOCK_ROWS, BLOCK_DIM_Z }; // 32 threads = 1 warp, N_BLOCK_ROWS warps per block for 16 rows, up to 2x 8 warps can share input (does not help vs. 1)
  const uint32_t n_elems_per_block = 16 * BLOCK_DIM_Z * N_ITERS;
  const uint32_t n_blocks = div_round_up(batch_size, n_elems_per_block);

  /* calculate shared memory size */
  constexpr uint32_t shmem_size_coord  = sizeof(float ) * (16 * BLOCK_DIM_Z * N_ITERS) * N_POS_DIMS;
  constexpr uint32_t shmem_size_output = sizeof(__half) * (16 * BLOCK_DIM_Z * N_ITERS) * 16;

  // 16*WIDTH rows of weights (for the last layer; others are in registers only) + 16*WIDTH*BLOCK_DIM_Z*N_ITERS rows of intermediate activations
  size_t shmem_size = sizeof(__half) * (16 + 16 * BLOCK_DIM_Z * N_ITERS) * (WIDTH + SKEW); 
  // If the input width is dynamic, the input weight matrix as well as part of the input will live in extra shared memory
  if (in_width != WIDTH) {
    shmem_size = std::max(shmem_size, sizeof(__half) * (WIDTH + 16 * BLOCK_DIM_Z) * (in_width + INPUT_SKEW));
  }
  shmem_size += shmem_size_coord + shmem_size_output;

  /* launch kernel */
  switch (mlp.activation) {
    case Activation::None:        break;
    case Activation::Exponential: break;
    case Activation::Sigmoid:     break;
    case Activation::ReLU:        break;
    case Activation::Squareplus:  break;
    case Activation::Softplus:    break;
    default: throw std::runtime_error{"Unsupported activation."};
  }
  const dim3 blocks = { n_blocks, 1u, 1u };
  check_shmem_error(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_size));

  TRACE_CUDA;

  kernel<<<blocks, threads, shmem_size, stream>>>(This, args...);

  // clang-format on

  TRACE_CUDA;
}

template<typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t WIDTH>
template<typename K, typename... Types>
void
DeviceNeuralVolume<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, WIDTH>::
launch1D(K kernel, cudaStream_t stream, int32_t width, Types... args) const
{
  if (width <= 0) { return; }
  launch_general(*this, kernel, stream, (uint32_t)width, width, args...);
}

template<typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t WIDTH>
template<typename K, typename... Types>
void
DeviceNeuralVolume<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, WIDTH>::
launch2D(K kernel, cudaStream_t stream, int32_t width, int32_t height, Types... args) const
{
  if (width <= 0 || height <= 0) { return; }
  launch_general(*this, kernel, stream, (uint32_t)width*height, width, height, args...);
}

template<typename T, uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL, uint32_t WIDTH>
template<typename K, typename... Types>
void
DeviceNeuralVolume<T, N_POS_DIMS, N_FEATURES_PER_LEVEL, WIDTH>::
launch3D(K kernel, cudaStream_t stream, int32_t width, int32_t height, int32_t depth, Types... args) const
{
  if (width <= 0 || height <= 0 || depth <= 0) { return; }
  launch_general(*this, kernel, stream, (uint32_t)width*height*depth, width, height, depth, args...);
}

constexpr static int TCNN_N_POS_DIMS = 3; // TODO find a better way

using DeviceVolume16x1 = DeviceNeuralVolume<precision_t, TCNN_N_POS_DIMS, 1, 16>;
using DeviceVolume16x2 = DeviceNeuralVolume<precision_t, TCNN_N_POS_DIMS, 2, 16>;
using DeviceVolume16x4 = DeviceNeuralVolume<precision_t, TCNN_N_POS_DIMS, 4, 16>;
using DeviceVolume16x8 = DeviceNeuralVolume<precision_t, TCNN_N_POS_DIMS, 8, 16>;
using DeviceVolume32x1 = DeviceNeuralVolume<precision_t, TCNN_N_POS_DIMS, 1, 32>;
using DeviceVolume32x2 = DeviceNeuralVolume<precision_t, TCNN_N_POS_DIMS, 2, 32>;
using DeviceVolume32x4 = DeviceNeuralVolume<precision_t, TCNN_N_POS_DIMS, 4, 32>;
using DeviceVolume32x8 = DeviceNeuralVolume<precision_t, TCNN_N_POS_DIMS, 8, 32>;
using DeviceVolume64x1 = DeviceNeuralVolume<precision_t, TCNN_N_POS_DIMS, 1, 64>;
using DeviceVolume64x2 = DeviceNeuralVolume<precision_t, TCNN_N_POS_DIMS, 2, 64>;
using DeviceVolume64x4 = DeviceNeuralVolume<precision_t, TCNN_N_POS_DIMS, 4, 64>;
using DeviceVolume64x8 = DeviceNeuralVolume<precision_t, TCNN_N_POS_DIMS, 8, 64>;

void tcnn_inference(network_t handler, cudaStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<float>& output);

}
}

namespace vnr {

template<int WIDTH, int N_FEATURES_PER_LEVEL>
using TcnnDeviceVolume = tcnn_impl::DeviceNeuralVolume<tcnn_impl::precision_t, tcnn_impl::TCNN_N_POS_DIMS, N_FEATURES_PER_LEVEL, WIDTH>;

} // namespace vnr
