
#pragma once

#include "api.h"

#include "core/instantvnr_types.h"
#include "core/network.h"
#include "core/sampler.h"

#include "serializer.h"
#include "renderer.h"

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
