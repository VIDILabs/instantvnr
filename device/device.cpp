// ----------------------------------------------------------------------------
//  device.cpp
//
//  Thin forwarders from `DeviceNNVolume` to `DeviceNNVolume::Impl` plus
//  the `OVR_REGISTER_OBJECT` line at the bottom of the file, which exposes
//  the class to OVR's dynamic object factory under the name `"nnvolume"`.
//  Host applications request it with `ovr::create_renderer("nnvolume")`.
// ----------------------------------------------------------------------------

#include "device.h"
#include "device_impl.h"

#include <ovr/common/dylink/ObjectFactory.h>

#include <chrono>

namespace ovr::nnvolume {

DeviceNNVolume::~DeviceNNVolume()
{
  pimpl.reset();
}

DeviceNNVolume::DeviceNNVolume() 
  : MainRenderer(), pimpl(new Impl()) 
{
}

void
DeviceNNVolume::init(int argc, const char** argv)
{
  pimpl->init(argc, argv, this);
  pimpl->commit();
}

void
DeviceNNVolume::swap()
{
}

void
DeviceNNVolume::commit()
{
  pimpl->commit();
}

void
DeviceNNVolume::render()
{
  auto start = std::chrono::high_resolution_clock::now();
  pimpl->render();
  auto end = std::chrono::high_resolution_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  render_time += diff.count();
}

void
DeviceNNVolume::mapframe(FrameBufferData* fb)
{
  return pimpl->mapframe(fb);
}

// void
// DeviceNNVolume::set_occlusion(vnrVolume v)
// {
//   // pimpl->set_occlusion(v);
// }

// void
// DeviceNNVolume::set_shadow(vnrVolume v)
// {
//   // pimpl->set_shadow(v);
// }

} // namespace ovr::nnvolume

OVR_REGISTER_OBJECT(ovr::MainRenderer, renderer, ovr::nnvolume::DeviceNNVolume, nnvolume)
