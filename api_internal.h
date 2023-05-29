
#pragma once

#include "api.h"

#include "core/types.h"
#include "core/serializer.h"
#include "core/renderer.h"
#include "core/network.h"
#include "core/sampler.h"

namespace vnr {

using namespace vnr::math;

struct VolumeContext 
{
  MultiVolume desc;
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

  bool isNetwork() const override { return true; };
};

struct RendererContext 
{
  MainRenderer renderer;
  vnrVolume volume;
};

}
