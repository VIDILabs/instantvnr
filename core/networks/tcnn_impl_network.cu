#include "tcnn_threadblock.h"
#include "tcnn_device_api.h"

// clang-format off

/* namespace instant neural volume */
namespace vnr {
namespace tcnn_impl {

template<int WIDTH, int BLOCK_DIM_Z, int N_ITERS, typename OUT_T>
__global__ void 
DeviceNeuralNetwork_batch_inference_kernel(
  const __half* __restrict__ input, 
  OUT_T* __restrict__ output, 
  const __half* __restrict__ weights, 
  const Activation ACTIVATION,
  const Activation output_activation,
  const uint32_t batch_size, 
  const uint32_t in_width,
  const uint32_t out_width,
  const uint32_t n_hidden_matmuls, 
  const nvcuda::wmma::layout_t input_layout, 
  const nvcuda::wmma::layout_t output_layout) 
{
  using namespace TCNN_NAMESPACE;

  // `input` points to the input matrix. Can be any width.
  // `weights` points to the weight matrices (contiguous in memory).
  // `output_intermediate` points to the memory where intermediate activations should be written. When performing inference, a value of nullptr is expected (intermediate results are not written).
  // `output` points to the memory where the network output should be written. (Output width is assumed to be 16 neurons.)

  // Commented out due to isolated strange side-effects on Windows
  // if (INFERENCE) {
  //   assert(output_intermediate == nullptr);
  // } else {
  //   assert(output_intermediate);
  // }

  // Shared memory contains the intermediate activations of blockDim.y*16 elements.
  // In some cases, it also contains the weight matrix for the first and last layer.
  extern __shared__ __half shmem[];
  __half* act_shmem = shmem;

  // Each block computes exactly one 16-element chunk of the batch.
  const uint32_t elem_idx = 16 * blockIdx.x * N_ITERS * BLOCK_DIM_Z;

  // First layer
  if (input_layout == nvcuda::wmma::mem_col_major || in_width != WIDTH) {
    if (input_layout == nvcuda::wmma::mem_row_major) {
      // CHANGE: only support inference
      // threadblock_input_layer_forward_dynamic<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T, nvcuda::wmma::row_major>(ACTIVATION, act_shmem, input + elem_idx * in_width, weights, !INFERENCE ? (out_intermediate + elem_idx * WIDTH) : nullptr, in_width, batch_size);
      threadblock_input_layer_forward_dynamic<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T, nvcuda::wmma::row_major>(ACTIVATION, act_shmem, input + elem_idx * in_width, weights, nullptr, in_width, batch_size);
    } else {
      // CHANGE: only support inference
      // threadblock_input_layer_forward_dynamic<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T, nvcuda::wmma::col_major>(ACTIVATION, act_shmem, input + elem_idx, weights, !INFERENCE ? (output_intermediate + elem_idx * WIDTH) : nullptr, in_width, batch_size);
      threadblock_input_layer_forward_dynamic<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T, nvcuda::wmma::col_major>(ACTIVATION, act_shmem, input + elem_idx, weights, nullptr, in_width, batch_size);
    }
  } else {
    // If the input has the same width & layout as the hidden layers, we can simply use the network's regular layer routine (with static size)
    // instead of using the slower dynamic input layer routine.
    threadblock_load_input_static<WIDTH, BLOCK_DIM_Z, N_ITERS>(act_shmem, input + elem_idx * WIDTH);
    // CHANGE: only support inference
    // threadblock_layer<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(ACTIVATION, act_shmem, weights, !INFERENCE ? (output_intermediate + elem_idx * WIDTH) : nullptr);
    threadblock_layer<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(ACTIVATION, act_shmem, weights, nullptr);
  }

  const uint32_t first_layer_size = WIDTH * in_width;
  const uint32_t layer_stride = WIDTH * WIDTH;
  const uint32_t output_stride = WIDTH * batch_size;

  // Hidden layers
  for (uint32_t k = 0; k < n_hidden_matmuls; ++k) {
    // CHANGE: only support inference
    // threadblock_layer<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(ACTIVATION, act_shmem, weights + first_layer_size + layer_stride * k, !INFERENCE ? (output_intermediate + output_stride * (k + 1) + elem_idx * WIDTH) : nullptr);
    threadblock_layer<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(ACTIVATION, act_shmem, weights + first_layer_size + layer_stride * k, nullptr);
  }

  // CHANGE: no need to support that many output layers
  assert(out_width <= 16 && "do not support more than 16 output layers");
  // if (out_width > 16) {
  //   // In the forward pass, intermediate activations are already written out.
  //   // if (INFERENCE) {
  //   //   threadblock_write_output_static<WIDTH, BLOCK_DIM_Z, N_ITERS>(act_shmem, output_intermediate + elem_idx * WIDTH);
  //   // }
  //  // CHANGE: only support inference
  //  threadblock_write_output_static<WIDTH, BLOCK_DIM_Z, N_ITERS>(act_shmem, output_intermediate + elem_idx * WIDTH);
  // } else if (output) {
  //   // Last layer
  //   if (output_layout == nvcuda::wmma::mem_row_major) {
  //     threadblock_last_layer_forward<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(output_activation, act_shmem, weights + first_layer_size + layer_stride * n_hidden_matmuls, output + elem_idx * 16, 16, output_layout);
  //   } else {
  //     threadblock_last_layer_forward<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(output_activation, act_shmem, weights + first_layer_size + layer_stride * n_hidden_matmuls, output + elem_idx, batch_size, output_layout);
  //   }
  // }

  // CHANGE: focus on inference only
  assert(output && "output cannot be NULL for inference");
  if (output_layout == nvcuda::wmma::mem_row_major) {
    threadblock_last_layer_forward<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(output_activation, act_shmem, weights + first_layer_size + layer_stride * n_hidden_matmuls, output + elem_idx * 16, 16, output_layout);
  } else {
    threadblock_last_layer_forward<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(output_activation, act_shmem, weights + first_layer_size + layer_stride * n_hidden_matmuls, output + elem_idx, batch_size, output_layout);
  }
}

template<typename T, int WIDTH>
void
DeviceNeuralNetwork<T, WIDTH>::batch_inference_internal(
  cudaStream_t stream, 
  Activation ACTIVATION,
  Activation output_activation,
  const GPUMatrix<T, RM>& weights,
  const uint32_t n_hidden_layers,
  const GPUMatrixDynamic<T>& input,
  GPUMatrixDynamic<T>& output) 
{
  static_assert(std::is_same<__half, T>::value, "only support half precision");

  using namespace TCNN_NAMESPACE;

  const uint32_t batch_size = input.cols();
  const uint32_t in_width = input.rows();
  const uint32_t out_width  = output.rows();

  constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0; // <- always going to be 8 as we only support multiple-of-16 widths
  constexpr uint32_t INPUT_SKEW = 8; // <- likewise with inputs
  constexpr uint32_t N_BLOCK_ROWS = WIDTH / 16;

  static_assert(WIDTH % 16 == 0, "Width must be a multiply of 16.");
  if (in_width % 16 != 0) {
    throw std::runtime_error{"Inputs must have a multiple-of-16 elements."};
  }

  if (weights.rows() != WIDTH) {
    throw std::runtime_error{"The fully fused forward pass only works with WIDTH-sized matrices."};
  }

  if (weights.cols() % 16 != 0) {
    throw std::runtime_error{std::string("weights must have a multiple-of-16 number of columns. ") + std::to_string(weights.cols())};
  }

  // CHANGE: output_intermediate is not needed for pure inferencing
  // if (output_intermediate.cols() != batch_size) {
  //   throw std::runtime_error{"Batch size of inputs and output_intermediate doesn't match."};
  // }

  // CHANGE: output should not be nullptr for inferencing
  if (/*output &&*/ output.cols() != batch_size) {
    throw std::runtime_error{"Batch size of inputs and outputs doesn't match."};
  }

  const int N_ITERS = WIDTH >= 256 ? 2 : 8;
  const uint32_t BLOCK_DIM_Z = (/*INFERENCE &&*/ WIDTH == 128) ? 2 : 1;

  if (batch_size % (16 * N_ITERS * BLOCK_DIM_Z) != 0) {
    throw std::runtime_error{"Batch size must be a multiple of " + std::to_string(16 * N_ITERS * BLOCK_DIM_Z) + "."};
  }

  const dim3 threads = { 32u, N_BLOCK_ROWS, BLOCK_DIM_Z }; // 32 threads = 1 warp, N_BLOCK_ROWS warps per block for 16 rows, up to 2x 8 warps can share input (does not help vs. 1)

  uint32_t n_elems_per_block = 16 * BLOCK_DIM_Z * N_ITERS;
  uint32_t n_blocks = div_round_up(batch_size, n_elems_per_block);

  size_t shmem_size = sizeof(__half) * (16 + 16 * BLOCK_DIM_Z * N_ITERS) * (WIDTH + SKEW); // 16*WIDTH rows of weights (for the last layer; others are in registers only) + 16*WIDTH*BLOCK_DIM_Z*N_ITERS rows of intermediate activations
  if (in_width != WIDTH || input.layout() == RM) {
    // If the input width is dynamic, the input weight matrix as well as part of the input will live in extra shared memory
    shmem_size = std::max(shmem_size, sizeof(__half) * (WIDTH + 16 * BLOCK_DIM_Z) * (in_width + INPUT_SKEW));
  }

  const dim3 blocks = { n_blocks, 1u, 1u };

  // CHANGE: simplify
  // check_shmem_error(cudaFuncSetAttribute(kernel_mlp_fused<WIDTH, BLOCK_DIM_Z, N_ITERS, __half, ACTIVATION, INFERENCE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_size));
  // kernel_mlp_fused<WIDTH, BLOCK_DIM_Z, N_ITERS, __half, ACTIVATION, INFERENCE><<<blocks, threads, shmem_size, stream>>>(
  //   output_activation, input.data(), weights.data(), 
  //  output_intermediate.data(), output ? output->data() : nullptr,
  //   batch_size, in_width,  output ? output->rows() : 0, n_hidden_layers,
  //   // The kernels operate with transposed layouts compared with the MLP code
  //   input.layout()             == RM ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major,
  //   output && output->layout() == RM ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major
  // );

  check_shmem_error(cudaFuncSetAttribute(DeviceNeuralNetwork_batch_inference_kernel<WIDTH, BLOCK_DIM_Z, N_ITERS, __half/*, ACTIVATION*/>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_size));
  DeviceNeuralNetwork_batch_inference_kernel<WIDTH, BLOCK_DIM_Z, N_ITERS, __half/*, ACTIVATION*/><<<blocks, threads, shmem_size, stream>>>(
    input.data(), output.data(), weights.data(), ACTIVATION, output_activation, batch_size, in_width, out_width, n_hidden_layers,
    // The kernels operate with transposed layouts compared with the MLP code
    input.layout()  == RM ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major,
    output.layout() == RM ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major
  );
}

template<typename T, int32_t WIDTH>
void
DeviceNeuralNetwork<T, WIDTH>::batch_inference(cudaStream_t stream, const Matrix<T>& input, Matrix<float>& _output)
{
  using namespace TCNN_NAMESPACE;

  // CHANGE: rename inference_output_tmp -> outputs
  GPUMatrixDynamic<T> output{ self->m_padded_output_width, _output.n(), stream, _output.layout() };

  // CHANGE: inline inference_mixed_precision
  // self->inference_mixed_precision(stream, input, output, true);
  
  // BEGIN INLINE
  
  // Various error checks
  if (input.m() != self->m_input_width)          throw std::runtime_error(std::string("Input has incorrect width: ") + std::to_string(input.m()) + "!=" + std::to_string(self->m_input_width));
  if (output.m() != self->m_padded_output_width) throw std::runtime_error(std::string("Output has incorrect width: ") + std::to_string(output.m()) + "!=" + std::to_string(self->m_output_width));
  if (input.n() != output.n())                   throw std::runtime_error(std::string("Input and output don't have matching batch size: ") + std::to_string(input.n()) + "!=" + std::to_string(output.n()));

  // Make sure our temporary buffers have the correct size for the given batch size
  // uint32_t batch_size = input.n();

  // CHANGE: immediate output is not needed for inference  
  // GPUMatrix<T> inference_tmp = self->m_output_width > 16 ? GPUMatrix<T>{self->m_network_width, batch_size, stream} : GPUMatrix<T>{nullptr, self->m_network_width, batch_size};
  // const WeightUsage weight_usage = use_inference_matrices ? WeightUsage::Inference : WeightUsage::Forward;
  //
  // ASSUMPTION: weight matrices are contiguous in memory
  // switch (self->m_activation) {
  //   case Activation::None:        mlp_fused_forward<WIDTH, T, Activation::None, true>(       stream, self->m_output_activation, self->input_weight_matrix(weight_usage), input, inference_tmp, &output, self->m_n_hidden_matmuls); break;
  //   case Activation::Exponential: mlp_fused_forward<WIDTH, T, Activation::Exponential, true>(stream, self->m_output_activation, self->input_weight_matrix(weight_usage), input, inference_tmp, &output, self->m_n_hidden_matmuls); break;
  //   case Activation::Sigmoid:     mlp_fused_forward<WIDTH, T, Activation::Sigmoid, true>(    stream, self->m_output_activation, self->input_weight_matrix(weight_usage), input, inference_tmp, &output, self->m_n_hidden_matmuls); break;
  //   case Activation::ReLU:        mlp_fused_forward<WIDTH, T, Activation::ReLU, true>(       stream, self->m_output_activation, self->input_weight_matrix(weight_usage), input, inference_tmp, &output, self->m_n_hidden_matmuls); break;
  //   case Activation::Sine:        mlp_fused_forward<WIDTH, T, Activation::Sine, true>(       stream, self->m_output_activation, self->input_weight_matrix(weight_usage), input, inference_tmp, &output, self->m_n_hidden_matmuls); break;
  //   case Activation::Squareplus:  mlp_fused_forward<WIDTH, T, Activation::Squareplus, true>( stream, self->m_output_activation, self->input_weight_matrix(weight_usage), input, inference_tmp, &output, self->m_n_hidden_matmuls); break;
  //   case Activation::Softplus:    mlp_fused_forward<WIDTH, T, Activation::Softplus, true>(   stream, self->m_output_activation, self->input_weight_matrix(weight_usage), input, inference_tmp, &output, self->m_n_hidden_matmuls); break;
  //   default: throw std::runtime_error{"Unsupported activation."};
  // }

  switch (self->m_activation) {
    case Activation::None:        batch_inference_internal(stream, Activation::None        ,self->m_output_activation, self->input_weight_matrix(WeightUsage::Inference), self->m_n_hidden_matmuls, input, output); break;
    case Activation::Exponential: batch_inference_internal(stream, Activation::Exponential ,self->m_output_activation, self->input_weight_matrix(WeightUsage::Inference), self->m_n_hidden_matmuls, input, output); break;
    case Activation::Sigmoid:     batch_inference_internal(stream, Activation::Sigmoid     ,self->m_output_activation, self->input_weight_matrix(WeightUsage::Inference), self->m_n_hidden_matmuls, input, output); break;
    case Activation::ReLU:        batch_inference_internal(stream, Activation::ReLU        ,self->m_output_activation, self->input_weight_matrix(WeightUsage::Inference), self->m_n_hidden_matmuls, input, output); break;
    case Activation::Sine:        batch_inference_internal(stream, Activation::Sine        ,self->m_output_activation, self->input_weight_matrix(WeightUsage::Inference), self->m_n_hidden_matmuls, input, output); break;
    case Activation::Squareplus:  batch_inference_internal(stream, Activation::Squareplus  ,self->m_output_activation, self->input_weight_matrix(WeightUsage::Inference), self->m_n_hidden_matmuls, input, output); break;
    case Activation::Softplus:    batch_inference_internal(stream, Activation::Softplus    ,self->m_output_activation, self->input_weight_matrix(WeightUsage::Inference), self->m_n_hidden_matmuls, input, output); break;
    default: throw std::runtime_error{"Unsupported activation."};
  }

  // If we have more than 16 output dimensions, these will be taken care of by CUTLASS rather than
  // the fully fused kernel (which will have written out the second-to-last layer activations).
  // CHANGE: no need to support that many output layers
  assert(self->m_output_width <= 16 && "do not support more than 16 output layers");
  // if (self->m_output_width > 16) {
  //   compute_inference_layer<LastLayer>(stream, self->m_output_activation, self->output_weight_matrix(weight_usage), inference_tmp, output);
  // }

  // END INLINE

  const uint32_t n_elements = (uint32_t)_output.n_elements();
  // If the layout is row major, trimming away excess dimensions amounts to simply discarding the tail of the buffer.
  if (_output.layout() == RM) {
    cast_from<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, output.data(), _output.data());
  }
  else {
    trim_and_cast<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, self->m_padded_output_width, self->m_output_width, output.data(), _output.data());
  }
}

template struct DeviceNeuralNetwork<precision_t, 16>;
template struct DeviceNeuralNetwork<precision_t, 32>;
template struct DeviceNeuralNetwork<precision_t, 64>;

}
}