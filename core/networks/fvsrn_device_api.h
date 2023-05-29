//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //
#pragma once

#include <cuda_runtime.h>

#include "../../../fvsrn/fvsrn_forward.h"
#include "../../../fvsrn/fvsrn_tensorcores_forward.h"

namespace vnr {

template<int WIDTH>
struct FvsrnDeviceVolume
{
  fvsrn::Defines defines;
  mutable kernel::SharedStorage storage;

  FvsrnDeviceVolume(void* ptr) : defines(*((fvsrn::Defines*)ptr)) {}

  __device__ void init() const;

  __device__ float sample(float3 coordinate) const;

  template<typename K, typename... Types>
  void launch(K kernel, cudaStream_t stream, uint32_t width, uint32_t height, Types... args) const;
};

template<int WIDTH>
template<typename K, typename... Types>
void
FvsrnDeviceVolume<WIDTH>::launch(K kernel, cudaStream_t stream, uint32_t width, uint32_t height, Types... args) const
{
  if (width <= 0 || height <= 0) { return; }

  TRACE_CUDA;

  const uint32_t batch_size = width * height; // = number of pixels
  const uint32_t block_size = defines.block_size;

  kernel<<<util::div_round_up<uint64_t>(batch_size, block_size), block_size, defines.shmem, stream>>>(*this, (uint32_t)width, (uint32_t)height, args...);

  TRACE_CUDA;
}

} // namespace vnr
