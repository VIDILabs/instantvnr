#pragma once
#ifndef OVR_NNVOLUME_DEVICE_H
#define OVR_NNVOLUME_DEVICE_H

#include "ovr/renderer.h"

#include "api.h"

#include <memory>

namespace ovr::nnvolume {

class DeviceNNVolume : public MainRenderer {
public:
  ~DeviceNNVolume() override;
  DeviceNNVolume();
  DeviceNNVolume(const DeviceNNVolume& other) = delete;
  DeviceNNVolume(DeviceNNVolume&& other) = delete;
  DeviceNNVolume& operator=(const DeviceNNVolume& other) = delete;
  DeviceNNVolume& operator=(DeviceNNVolume&& other) = delete;

  /*! constructor - performs all setup, including initializing ospray, creates scene graph, etc. */
  void init(int argc, const char** argv) override;

  /*! render one frame */
  void swap() override;
  void commit() override;
  void render() override;
  void mapframe(FrameBufferData* fb) override;

  // void set_occlusion(vnrVolume);
  // void set_shadow(vnrVolume);

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl; // pointer to the internal implementation
};

} // namespace ovr::nnvolume

#endif // OVR_NNVOLUME_DEVICE_H
