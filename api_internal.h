
#pragma once

#include "api.h"

#include "core/instantvnr_types.h"
#include "core/network.h"
#include "core/sampler.h"
#include "core/renderer.h"
#include "core/serializer.h"
#include "core/framebuffer.h"

namespace vnr {

using namespace vnr::math;

struct VolumeContext 
{
  vec3i     dims;
  ValueType type;
  range1f   range;
  box3f clipbox;
  virtual ~VolumeContext() {};
  virtual bool isNetwork() const = 0;
};

struct SimpleVolumeContext : VolumeContext 
{
  SimpleVolume source;

  bool isNetwork() const override { return false; };
};

struct NeuralVolumeContext : VolumeContext 
{
  NeuralVolume neural;
  NeuralVolume::Statistics stats;

  NeuralVolumeContext(size_t batchsize) : neural(batchsize) {}

  bool isNetwork() const override { return true; };
};

struct RenderContext
{
  vnrVolume volume;
  Camera camera;
  TransferFunctionAPI tfn;
  RenderAPI render;

  // other states
  int rendering_mode{ VNR_INVALID };

  // volume states
  float sampling_rate{ 1.f };
  float density_scale{ 1.f };

  // framebuffer states
  FrameBuffer framebuffer;
  cudaStream_t framebuffer_stream{};
  vec2i framebuffer_size;
  bool framebuffer_reset{ true };
};

}
