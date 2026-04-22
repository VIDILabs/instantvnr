// ----------------------------------------------------------------------------
//  fvsrn_network.h
//
//  Alternative network backend (fV-SRN, "fast volume scene representation
//  network") selected when the library is built with `ENABLE_FVSRN=ON`.
//  Implements the same `AbstractNetwork` contract as `TcnnNetwork` so
//  `NeuralVolume` can be swapped between backends at construction time.
//
//  Enabling fV-SRN forces `DISABLE_ADAPTIVE_SAMPLING=ON` and
//  `KERNEL_DOUBLE_PRECISION=1` (see `core/CMakeLists.txt`) because the
//  fV-SRN kernels do not support macrocell acceleration.
// ----------------------------------------------------------------------------

#pragma once

#include "tcnn_network.h"

#include <iostream>
#include <memory>

namespace vnr {

class FvsrnNetwork : public AbstractNetwork
{
  struct Impl;
  std::unique_ptr<Impl> pimpl; // pointer to the internal implementation

public:
  FvsrnNetwork();
  ~FvsrnNetwork();

  int n_input_dims() const { return 3; }
  int n_output_dims() const { return 1; }
  int n_neurons() const;
  int n_features_per_level() const { return -2; }

  bool valid() const { return true; }

  void* network_direct_access();

  size_t get_model_size() const { return 0; }

  size_t get_mlp_size() const { return 0; }

  size_t get_enc_size() const { return 0; }

  size_t training_step() const { return 0; }

  double training_loss() const { return -1.0; }

  json serialize_model() const {
    throw std::runtime_error("[fvsrn] fV-SRN network model cannot be serialized");
  }

  void deserialize_model(json config);

  json serialize_params() const { 
    throw std::runtime_error("[fvsrn] fV-SRN network parameters cannot be serialized");
  }

  void deserialize_params(const json& parameters);

  void train(const GPUColumnMatrix& input, const GPUColumnMatrix& target, cudaStream_t stream) {
    throw std::runtime_error("[fvsrn] fV-SRN network cannot be trained");
  }

  void infer(const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<float>& output, cudaStream_t stream) const;
};

}
