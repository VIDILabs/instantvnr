#pragma once

#include "../object.h"

#include <cuda/cuda_buffer.h>

namespace vnr {

class MethodPathTracing
{
public:
  void render(cudaStream_t stream, const LaunchParams& params, StructuredRegularVolume& volume, NeuralVolume* nvr = nullptr, bool iterative = false);
  void clear(cudaStream_t stream) { sample_streaming_buffer.free(stream); }

private:
  CUDABuffer sample_streaming_buffer;
};

} // namespace vnr
