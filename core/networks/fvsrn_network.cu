#include <string>
#include <vector>
#include <random>
#include <fstream>

#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <mma.h>

#include "fvsrn_network.h"
#include "fvsrn_device_api.h"

// defined in the model (should use JIT instead)
#include "../../../fvsrn/model_defs.inl"

#include "../../../fvsrn/volume_interpolation_network.h"

namespace fvsrn {

renderer::GlobalSettings global_settings()
{
	renderer::GlobalSettings s{};
	s.volumeShouldProvideNormals = false;
	s.interpolationInObjectSpace = false;
  return s;
}

template<int WIDTH_DIV16>
__global__ void SRNInfer(uint32_t N, const float* __restrict__ coords, float* __restrict__ values, fvsrn::Defines defs)
{
	kernel::VolumeInterpolationTensorcores<WIDTH_DIV16> srn(defs);

  const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

	auto in  = make_real3(coords[3*i+0], coords[3*i+1], coords[3*i+2]);
	auto out = srn.template eval<real_t>(in, dummy_direction, 0);

	values[i] = out.value;
}

}

namespace vnr {

template<int WIDTH> 
__device__ void 
FvsrnDeviceVolume<WIDTH>::init() const 
{
	((kernel::VolumeInterpolationTensorcores<WIDTH/16>&)storage).load(defines);
}

template<int WIDTH> 
__device__ float 
FvsrnDeviceVolume<WIDTH>::sample(const float3 coordinate) const
{
	auto& srn = ((kernel::VolumeInterpolationTensorcores<WIDTH/16>&)storage);
  return srn.template eval<real_t>(make_real3(coordinate.x, coordinate.y, coordinate.z), dummy_direction, 0).value;
}

template struct FvsrnDeviceVolume<16*1>;
template struct FvsrnDeviceVolume<16*2>;
template struct FvsrnDeviceVolume<16*3>;
template struct FvsrnDeviceVolume<16*4>;
template struct FvsrnDeviceVolume<16*5>;
template struct FvsrnDeviceVolume<16*6>;
template struct FvsrnDeviceVolume<16*7>;
template struct FvsrnDeviceVolume<16*8>;

struct FvsrnNetwork::Impl {
	renderer::VolumeInterpolationNetwork net;
  renderer::SceneNetwork_ptr model;
  fvsrn::Defines defines;
	CUDABuffer constant; 
};

FvsrnNetwork::FvsrnNetwork() : pimpl(new Impl()) {}

FvsrnNetwork::~FvsrnNetwork() { pimpl.reset(); }

int 
FvsrnNetwork::FUSED_MLP_WIDTH() const { return pimpl->defines.hidden_channels_div16 * 16; }

void* 
FvsrnNetwork::network_direct_access() { return &pimpl->defines; }

void
FvsrnNetwork::deserialize_params(const json& config)
{
	renderer::GlobalSettings s = fvsrn::global_settings();

  // auto volnet = config["model"]["fvsrn"].get<std::string>();
	// pimpl->net.loadNetwork(volnet);

	json::binary_t volnet = config["params_binary"];

	std::stringstream iss;
	iss.write((const char*)volnet.data(), volnet.size());

	pimpl->net.addNetwork(renderer::SceneNetwork::load(iss), "");
	pimpl->net.setBoxMin(make_double3(0, 0, 0));
	pimpl->net.setBoxMax(make_double3(1, 1, 1));
	pimpl->net.prepareRendering(s);

	std::cout << "-------" << std::endl;
	std::cout << pimpl->net.getDefines(s, pimpl->defines) << std::endl;
	std::cout << "-------" << std::endl;

	// verify parameters
	fvsrn::verify(pimpl->defines);

  // setup params & upload parameters
	pimpl->defines.shmem = kernel::SharedStorage::nbytes(pimpl->defines);

#ifndef DISABLE_CONSTANT_MEMORY
  cudaGetSymbolAddress(&pimpl->defines.constant, volumeInterpolationTensorcoresParameters);
#else
  pimpl->constant.alloc(kernel::ConstStorage::nbytes(pimpl->defines), 0);
  pimpl->defines.constant = (void*)pimpl->constant.d_pointer();
#endif

	pimpl->net.fillConstantMemory(s, (CUdeviceptr)pimpl->defines.constant, 0);

	CUDA_SYNC_CHECK();
}

void 
FvsrnNetwork::deserialize_model(json config)
{
  TRACE_CUDA;
  TRACE_CUDA;
}

void
FvsrnNetwork::infer(const GPUMatrixDynamic<float>& coord, GPUMatrixDynamic<float>& output, cudaStream_t stream) const
{
  TRACE_CUDA;

  assert(coord.layout() == TCNN_NAMESPACE :: MatrixLayout::ColumnMajor && "input coordinate should be a column major matrix");
  assert(coord.m() == 3 && "incorrect coordinate buffer shape");
  assert(output.n() == coord.n() && "incorrect output buffer shape");
  assert(output.m() == 1 && "incorrect coordinate buffer shape");

  const uint32_t batch_size = coord.n();
	const uint32_t block_size = pimpl->defines.block_size;
	const uint32_t shmem = pimpl->defines.shmem;

	if      (pimpl->defines.hidden_channels_div16 == 1) fvsrn::SRNInfer<1><<<util::div_round_up<uint64_t>(batch_size, block_size), block_size, shmem, stream>>>(batch_size, (float*)coord.data(), (float*)output.data(), pimpl->defines);
	else if (pimpl->defines.hidden_channels_div16 == 2) fvsrn::SRNInfer<2><<<util::div_round_up<uint64_t>(batch_size, block_size), block_size, shmem, stream>>>(batch_size, (float*)coord.data(), (float*)output.data(), pimpl->defines);
	else if (pimpl->defines.hidden_channels_div16 == 3) fvsrn::SRNInfer<3><<<util::div_round_up<uint64_t>(batch_size, block_size), block_size, shmem, stream>>>(batch_size, (float*)coord.data(), (float*)output.data(), pimpl->defines);
	else if (pimpl->defines.hidden_channels_div16 == 4) fvsrn::SRNInfer<4><<<util::div_round_up<uint64_t>(batch_size, block_size), block_size, shmem, stream>>>(batch_size, (float*)coord.data(), (float*)output.data(), pimpl->defines);
	else if (pimpl->defines.hidden_channels_div16 == 5) fvsrn::SRNInfer<5><<<util::div_round_up<uint64_t>(batch_size, block_size), block_size, shmem, stream>>>(batch_size, (float*)coord.data(), (float*)output.data(), pimpl->defines);
	else if (pimpl->defines.hidden_channels_div16 == 6) fvsrn::SRNInfer<6><<<util::div_round_up<uint64_t>(batch_size, block_size), block_size, shmem, stream>>>(batch_size, (float*)coord.data(), (float*)output.data(), pimpl->defines);
	else if (pimpl->defines.hidden_channels_div16 == 7) fvsrn::SRNInfer<7><<<util::div_round_up<uint64_t>(batch_size, block_size), block_size, shmem, stream>>>(batch_size, (float*)coord.data(), (float*)output.data(), pimpl->defines);
	else if (pimpl->defines.hidden_channels_div16 == 8) fvsrn::SRNInfer<8><<<util::div_round_up<uint64_t>(batch_size, block_size), block_size, shmem, stream>>>(batch_size, (float*)coord.data(), (float*)output.data(), pimpl->defines);
	else throw std::runtime_error("too many hidden channels");

  TRACE_CUDA;
}

}
