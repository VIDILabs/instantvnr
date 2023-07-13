#pragma once

#include "../instantvnr_types.h"
#include "../network.h"

#include <cuda/cuda_buffer.h>


INSTANT_VNR_NAMESPACE_BEGIN

class MethodRayMarching
{
public:
  enum ShadingMode { NO_SHADING = 0, GRADIENT_SHADING, SINGLE_SHADE_HEURISTIC, SHADOW };

  ~MethodRayMarching() { clear(0); }
  void render(cudaStream_t stream, const LaunchParams& params, ShadingMode mode, DeviceVolume* volume, NeuralVolume* nvr = nullptr, bool iterative = false);
  void clear(cudaStream_t stream) { sample_streaming_buffer.free(stream); }

private:
  CUDABuffer sample_streaming_buffer;
};

INSTANT_VNR_NAMESPACE_END
