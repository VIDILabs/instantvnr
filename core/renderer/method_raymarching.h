#pragma once

#include "../object.h"

#include <cuda/cuda_buffer.h>

namespace vnr {

class MethodRayMarching
{
public:
  enum ShadingMode { NO_SHADING = 0, GRADIENT_SHADING, SINGLE_SHADE_HEURISTIC, SHADOW };

  ~MethodRayMarching() { clear(0); }
  void render(cudaStream_t stream, const LaunchParams& params, StructuredRegularVolume& volume, ShadingMode mode, NeuralVolume* nvr = nullptr, bool iterative = false);
  void clear(cudaStream_t stream) { sample_streaming_buffer.free(stream); }

private:
  CUDABuffer sample_streaming_buffer;
};

} // namespace vnr
