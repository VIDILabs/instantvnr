#pragma once

#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/gpu_memory_json.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>

#include <json/json.hpp>

#include "../instantvnr_types.h"

#define VNR_INPUT_DIMS 3
#define VNR_OUTPUT_DIMS 1

#define TCNN_NEW_API

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
using Trainer                  = TCNN_NAMESPACE :: Trainer<float, precision_t, precision_t>;
using NetworkWithInputEncoding = TCNN_NAMESPACE :: NetworkWithInputEncoding<precision_t>;

using TCNN_NAMESPACE :: create_loss;
using TCNN_NAMESPACE :: create_optimizer;

using network_t = std::shared_ptr<NetworkWithInputEncoding>;

}
}

// ------------------------------------------------------------------
// Public Interface
// ------------------------------------------------------------------

namespace vnr {

struct AbstractNetwork {
  virtual ~AbstractNetwork() {}

  virtual int n_input_dims() const = 0;
  virtual int n_output_dims() const = 0;
  virtual int n_neurons() const = 0;
  virtual int n_features_per_level() const = 0;
  
  virtual bool valid() const = 0;
  
  virtual void* network_direct_access() = 0;
  
  virtual size_t get_model_size() const = 0;
  
  virtual size_t training_step() const = 0;
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

#ifdef TCNN_NEW_API
  std::unique_ptr<tcnn_impl::Trainer::ForwardContext> m_ctx;
#endif

  // training parameters
  uint64_t m_training_step = 0;
#ifndef TCNN_NEW_API
  mutable double   m_training_loss = 0;
  mutable uint64_t m_training_loss_count = 0;
#endif

  json m_model;
  json m_optimizer_opts = json::object();
  
  int N_NEURONS = -1;
  int N_FEATURES_PER_LEVEL = -1;

public:
  int n_input_dims() const { return INPUT_SIZE; }
  int n_output_dims() const { return OUTPUT_SIZE; }
  int n_neurons() const { return N_NEURONS; }
  int n_features_per_level() const { return N_FEATURES_PER_LEVEL; }

  bool valid() const { return m_trainer.get() != nullptr; }

  void* network_direct_access() { return m_network.get(); }
  
  size_t get_model_size() const { return sizeof(tcnn_impl::precision_t) * m_network->n_params(); }

  size_t training_step() const { return m_training_step; }

  double training_loss() const {
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

  void deserialize_model(json config) {
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
      N_NEURONS = network_opts["n_neurons"].get<int>();
      logging() << "[network] WIDTH = " << N_NEURONS << std::endl;
    }
    else {
      N_NEURONS = -1;
      logging() << "[network] other MLP format" << std::endl;
    }

    if (encoding_opts["otype"] == "HashGrid") {
      N_FEATURES_PER_LEVEL = encoding_opts["n_features_per_level"].get<int>();
      logging() << "[network] N_FEATURES_PER_LEVEL = " << N_FEATURES_PER_LEVEL << std::endl;
    }
    else {
      N_FEATURES_PER_LEVEL = -1;
      logging() << "[network] other encoding method" << std::endl;
    }

    m_loss.reset();
    m_optimizer.reset();
    m_network.reset();
    m_trainer.reset();

    try {
        m_loss = std::shared_ptr<Loss>{ create_loss<precision_t>(loss_opts) };
        m_optimizer = std::shared_ptr<Optimizer>{ create_optimizer<precision_t>(optimizer_opts) };
        m_network = std::make_shared<NetworkWithInputEncoding>(INPUT_SIZE, OUTPUT_SIZE, encoding_opts, network_opts);
        m_trainer = std::make_shared<Trainer>(m_network, m_optimizer, m_loss, (uint32_t)time(NULL));
    }
    catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
    }

    m_training_step = 0;
#ifndef TCNN_NEW_API
    m_training_loss = 0;
    m_training_loss_count = 0;
#endif
    logging() << "[network] total # of parameters = " << m_network->n_params() << std::endl;
  }

  void train(const GPUColumnMatrix& input, const GPUColumnMatrix& target, cudaStream_t stream) {
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
    ++m_training_step;

    TRACE_CUDA;
  }

  void infer(const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<float>& output, cudaStream_t stream) const {
    TRACE_CUDA;

    try {
      m_network->inference(stream, input, output);
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
