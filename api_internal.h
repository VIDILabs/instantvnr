// ----------------------------------------------------------------------------
//  api_internal.h
//
//  Internal composition layer shared between `api.cpp` and the OVR plugin in
//  `device/`. NOT installed; NOT part of the public ABI. Downstream consumers
//  should include `api.h` instead.
//
//  This header stitches the public `vnr*` handle typedefs together with the
//  concrete core types: a `VolumeContext` either wraps a `SimpleVolume`
//  (ground-truth data on disk / in memory) or a `NeuralVolume` (tiny-cuda-nn
//  or fV-SRN backed trainable representation), and a `RenderContext` ties a
//  volume to its camera, transfer function, framebuffer, and active
//  `vnrRenderMode`.
// ----------------------------------------------------------------------------

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
