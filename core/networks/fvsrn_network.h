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

  int n_input()  const { return 3; }
  int n_output() const { return 1; }

  int FUSED_MLP_WIDTH()      const;
  int N_FEATURES_PER_LEVEL() const { return -2; }

  bool valid() const { return true; }

  double training_loss() const { return -1.0; }
  
  size_t get_model_size() const { return 0; }

  size_t steps() const { return 0; }

  void* network_direct_access();

  json serialize_params() const 
  { 
    throw std::runtime_error("[fvsrn] fV-SRN network parameters cannot be serialized");
  }

  json serialize_model() const 
  {
    throw std::runtime_error("[fvsrn] fV-SRN network model cannot be serialized");
  }

  void train(const GPUColumnMatrix& input, const GPUColumnMatrix& target, cudaStream_t stream)
  {
    throw std::runtime_error("[fvsrn] fV-SRN network cannot be trained");
  }

  void deserialize_model(json config);

  void deserialize_params(const json& parameters);

  void infer(const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<float>& output, cudaStream_t stream) const;
};

}
