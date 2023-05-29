#pragma once

#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/gpu_memory_json.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/encodings/grid.h>

#include <json/json.hpp>

#include "../types.h"

#ifndef ENABLE_IN_SHADER // TODO automatically detect the new API
#define TCNN_NEW_API
#endif


#ifdef ENABLE_LOGGING
#define logging() std::cout
#else
static std::ostream null_output_stream(0);
#define logging() null_output_stream
#endif

// ------------------------------------------------------------------
// Shared Definitions
// ------------------------------------------------------------------

namespace vnr {

using json = nlohmann::json;

using TCNN_NAMESPACE :: GPUMatrix;
using TCNN_NAMESPACE :: GPUMemory;
using TCNN_NAMESPACE :: GPUMatrixDynamic;

using GPUColumnMatrix = TCNN_NAMESPACE :: GPUMatrix<float, TCNN_NAMESPACE :: MatrixLayout::ColumnMajor>;
using GPURowMatrix    = TCNN_NAMESPACE :: GPUMatrix<float, TCNN_NAMESPACE :: MatrixLayout::RowMajor>;

using TCNN_NAMESPACE :: json_binary_to_gpu_memory;
using TCNN_NAMESPACE :: gpu_memory_to_json_binary;

}

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

namespace vnr {
namespace tcnn_impl {

using precision_t = TCNN_NAMESPACE :: network_precision_t;

using Loss                     = TCNN_NAMESPACE :: Loss<precision_t>;
using Optimizer                = TCNN_NAMESPACE :: Optimizer<precision_t>;
using NetworkWithInputEncoding = TCNN_NAMESPACE :: NetworkWithInputEncoding<precision_t>;
using Trainer                  = TCNN_NAMESPACE :: Trainer<float, precision_t, precision_t>;
using Encoding                 = TCNN_NAMESPACE :: Encoding<precision_t>;

using TCNN_NAMESPACE :: create_loss;
using TCNN_NAMESPACE :: create_optimizer;
using TCNN_NAMESPACE :: create_grid_encoding;

using network_t = std::shared_ptr<NetworkWithInputEncoding>;

void tcnn_inference(network_t handler, cudaStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<float>& output);

}
}

// ------------------------------------------------------------------
// Public Interface
// ------------------------------------------------------------------

namespace vnr {

struct AbstractNetwork {
  virtual ~AbstractNetwork() {}
  virtual int n_input() const = 0;
  virtual int n_output() const = 0;
  virtual int FUSED_MLP_WIDTH() const = 0;
  virtual int N_FEATURES_PER_LEVEL() const = 0;
  virtual bool valid() const = 0;
  virtual size_t get_model_size() const = 0;
  virtual size_t steps() const = 0;
  virtual void* network_direct_access() = 0;
  virtual double training_loss() const = 0;
  virtual json serialize_params() const = 0;
  virtual void deserialize_params(const json& parameters) = 0;
  virtual json serialize_model() const = 0;
  virtual void deserialize_model(json config) = 0;
  virtual void train(const GPUColumnMatrix& input, const GPUColumnMatrix& target, cudaStream_t stream) = 0;
  virtual void infer(const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<float>& output, cudaStream_t stream) const = 0;
};

template<int INPUT_SIZE, int OUTPUT_SIZE>
struct TcnnNetwork : AbstractNetwork
{
  using network_t = tcnn_impl::network_t;

private:
  mutable std::shared_ptr<tcnn_impl::Loss> m_loss;
  mutable std::shared_ptr<tcnn_impl::Optimizer> m_optimizer;
  mutable std::shared_ptr<tcnn_impl::NetworkWithInputEncoding> m_network;
  mutable std::shared_ptr<tcnn_impl::Trainer> m_trainer;
#ifndef ENABLE_IN_SHADER
  mutable std::shared_ptr<tcnn_impl::Encoding> m_encoder;
#endif

#ifdef TCNN_NEW_API
  std::unique_ptr<tcnn_impl::Trainer::ForwardContext> m_ctx;
#endif

  // training parameters
  uint64_t m_steps = 0;

#ifndef TCNN_NEW_API
  mutable double   m_training_loss = 0;
  mutable uint64_t m_training_loss_count = 0;
#endif

  json m_model;
  json m_optimizer_opts = json::object();
  int network_FUSED_MLP_WIDTH = -1;
  int network_N_FEATURES_PER_LEVEL = -1;

public:
  int n_input() const { return INPUT_SIZE; }
  int n_output() const { return OUTPUT_SIZE; }

  int FUSED_MLP_WIDTH() const { return network_FUSED_MLP_WIDTH; }
  int N_FEATURES_PER_LEVEL() const { return network_N_FEATURES_PER_LEVEL; }

  bool valid() const { return m_trainer.get() != nullptr; }

  size_t get_model_size() const { return sizeof(tcnn_impl::precision_t) * m_network->n_params(); }

  size_t steps() const { return m_steps; }

  void* network_direct_access() { return m_network.get(); }

  double training_loss() const
  {
#ifdef TCNN_NEW_API
    return m_trainer->loss(0, *m_ctx);
#else
    if (m_training_loss_count > 0) {
      m_training_loss = m_training_loss / (double)m_training_loss_count;
      m_training_loss_count = 0;
    }
    return m_training_loss;
#endif
  }

  json serialize_params() const { return m_trainer->serialize(); }

  void deserialize_params(const json& parameters) { m_trainer->deserialize(parameters); }

  json serialize_model() const { return m_model; }

  void deserialize_model(json config)
  {
    using namespace tcnn_impl;

    json loss_opts = config.value("loss", json::object());
    json encoding_opts = config.value("encoding", json::object());
    json network_opts = config.value("network", json::object());
    json optimizer_opts = config.value("optimizer", m_optimizer_opts);

    m_model["loss"] = loss_opts;
    m_model["encoding"] = encoding_opts;
    m_model["network"] = network_opts;
    m_optimizer_opts = optimizer_opts;

    if (network_opts["otype"] == "FullyFusedMLP") {
        network_FUSED_MLP_WIDTH = network_opts["n_neurons"].get<int>();
        logging() << "[network] WIDTH = " << network_FUSED_MLP_WIDTH << std::endl;
    }
    else {
        network_FUSED_MLP_WIDTH = -1;
        logging() << "[network] other MLP format" << std::endl;
    }

    if (encoding_opts["otype"] == "HashGrid") {
        network_N_FEATURES_PER_LEVEL = encoding_opts["n_features_per_level"].get<int>();
        logging() << "[network] N_FEATURES_PER_LEVEL = " << network_N_FEATURES_PER_LEVEL << std::endl;
    }
    else {
        network_N_FEATURES_PER_LEVEL = -1;
        logging() << "[network] other encoding method" << std::endl;
    }

    m_loss.reset();
    m_optimizer.reset();
    m_network.reset();
    m_trainer.reset();

    try {
        m_loss = std::shared_ptr<Loss>{ create_loss<precision_t>(loss_opts) };
        m_optimizer = std::shared_ptr<Optimizer>{ create_optimizer<precision_t>(optimizer_opts) };
#ifdef ENABLE_IN_SHADER
        m_network = std::make_shared<NetworkWithInputEncoding>(INPUT_SIZE, OUTPUT_SIZE, encoding_opts, network_opts);
#else
        m_encoder = std::shared_ptr<Encoding>{ create_grid_encoding<precision_t>(INPUT_SIZE, encoding_opts) };
        m_network = std::make_shared<NetworkWithInputEncoding>(m_encoder, OUTPUT_SIZE, network_opts);
#endif
        m_trainer = std::make_shared<Trainer>(m_network, m_optimizer, m_loss, (uint32_t)time(NULL));
    }
    catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
    }

    m_steps = 0;
#ifndef TCNN_NEW_API
    m_training_loss = 0;
    m_training_loss_count = 0;
#endif
    logging() << "[network] total # of parameters = " << m_network->n_params() << std::endl;
  }

  void train(const GPUColumnMatrix& input, const GPUColumnMatrix& target, cudaStream_t stream)
  {
    TRACE_CUDA;

    float loss;

    try {
#ifdef TCNN_NEW_API
        m_ctx = m_trainer->training_step(stream, input, target);
#else
        m_trainer->training_step(stream, input, target, &loss);
#endif   
    }
    catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        m_loss.reset();
        m_optimizer.reset();
        m_network.reset();
        m_trainer.reset();
        return;
    }

#ifndef TCNN_NEW_API
    m_training_loss += loss;
    ++m_training_loss_count;
#endif
    ++m_steps;

    TRACE_CUDA;
  }

  void infer(const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<float>& output, cudaStream_t stream) const
  {
    TRACE_CUDA;

    try {
      tcnn_impl::tcnn_inference(m_network, stream, input, output);
    }
    catch (std::runtime_error& e) {
      std::cerr << e.what() << std::endl;
      m_loss.reset();
      m_optimizer.reset();
      m_network.reset();
      m_trainer.reset();
      return;
    }

    TRACE_CUDA;
  }
};

}
