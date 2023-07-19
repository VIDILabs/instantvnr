#include "sampler.h"
#include "samplers/neural_sampler.h"

INSTANT_VNR_NAMESPACE_BEGIN

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

Sampler
SamplerAPI::create(const MultiVolume& desc, std::string training_mode, bool save_volume)
{
  Sampler impl;

  /* GPU-based */
  if (training_mode == "GPU") {
    impl = std::make_shared<CudaSampler_TimeVarying>(desc, save_volume, false);
    impl->m_rendering_dims = impl->dims();
  }

#ifdef ENABLE_OUT_OF_CORE

  /* CPU-based, virtual memory, no ground truth */
  else if (training_mode == "VIRTUAL_MEMORY") {
    impl = std::make_shared<VirtualMemorySampler>(desc);
    impl->m_rendering_dims = gdt::min(vec3i(1024), vec3i(desc.dims));
  }

  /* out-of-core-steaming */
  else if (training_mode == "OUT_OF_CORE") {
    impl = std::make_shared<OutOfCoreSampler>(desc);
    impl->m_rendering_dims = gdt::min(vec3i(1024), vec3i(desc.dims));
  }

#endif // ENABLE_OUT_OF_CORE

#ifdef ENABLE_OPENVKL

  /* CPU-based, openvkl, no ground truth */
  else if (training_mode == "OPENVKL") {
    impl = std::make_shared<OpenVKLSampler>(desc.data[0], desc.dims, desc.type, desc.range);
    impl->m_rendering_dims = gdt::min(vec3i(1024), vec3i(desc.dims));
  }

  /* use OpenVKL sampling but with a ground truth at the original resolution */
  else if (training_mode == "OPENVKL_GT_ORIGINAL_RESOLUTION") {
    impl = std::make_shared<OpenVKLSampler_WithGroundTruthData>(desc);
    impl->m_rendering_dims = impl->dims();
  }

  /* use OpenVKL sampling but with a ground truth at the original resolution */
  else if (training_mode == "OPENVKL_GT_DOWNSAMPLE_RESOLUTION") {
    impl = std::make_shared<OpenVKLSampler_WithGroundTruthData>(desc, vec3i(desc.dims) / 8); // downsampled by 8x
    impl->m_rendering_dims = impl->dims();
  }

  /* using OpenVKL to support irregular datasets */
  else if (training_mode == "OPENVKL_IRREGULAR") {
    impl = std::make_shared<OpenVKLSampler>("WaveletVdb");
    impl->m_rendering_dims = vec3i(impl->dims()) * 1024 / gdt::reduce_max(vec3i(impl->dims()));
    impl->m_transform = affine3f::translate(-vec3f(impl->m_rendering_dims) / 2.f) * affine3f::scale(vec3f(impl->m_rendering_dims));
    return impl;
  }

  else if (training_mode.substr(0,11) == "OPENVKL_VDB") {
    impl = std::make_shared<OpenVKLSampler>("data/" + training_mode.substr(13) + ".vdb", "density");
    impl->m_rendering_dims = vec3i(impl->dims()) * 1024 / gdt::reduce_max(vec3i(impl->dims()));
    impl->m_transform = affine3f::translate(-vec3f(impl->m_rendering_dims) / 2.f) * affine3f::scale(vec3f(impl->m_rendering_dims));
    return impl;
  }

#endif // ENABLE_OPENVKL

  else if (training_mode == "NOTHING") {
    impl = std::make_shared<DummySampler>(desc.dims);
    impl->m_rendering_dims = impl->dims();
  }

  else throw std::runtime_error("unknown mode: " + training_mode);

  impl->m_transform = affine3f::translate(vec3f(desc.dims) * -0.5f) * affine3f::scale(vec3f(desc.dims));
  return impl;
}

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

void
SimpleVolume::load(const MultiVolume& descriptor, std::string sampling_mode, bool save_volume)
{
  desc = descriptor;
  mode = sampling_mode;
  sampler = SamplerAPI::create(desc, sampling_mode, save_volume);
  tex = sampler->texture();
  if (tex) {
    macrocell.set_shape(desc.dims);
    macrocell.allocate();
    macrocell.compute_everything(tex);
  }
}

void 
SimpleVolume::load(const void* data, vec3i dims, std::string type, range1f range, std::string sampling_mode)
{
  mode = sampling_mode;

  desc.dims = dims;
  desc.type = value_type(type);
  desc.range = range;

  sampler = std::make_shared<CudaSampler>(data, desc.dims, desc.type, desc.range, true);
  sampler->set_rendering_dims(sampler->dims());
  sampler->set_transform(affine3f::translate(vec3f(desc.dims) * -0.5f) * affine3f::scale(vec3f(desc.dims)));

  tex = sampler->texture();
  if (tex) {
    macrocell.set_shape(desc.dims);
    macrocell.allocate();
    macrocell.compute_everything(tex);
  }
}

void 
SimpleVolume::set_current_timestep(int index) 
{ 
  sampler->set_current_volume_index(index); 
  if (tex && !macrocell.is_external()) {
    macrocell.compute_everything(tex);
  }
}

void SimpleVolume::set_transfer_function(const std::vector<vec3f>& c, const std::vector<vec2f>& o, const range1f& r)
{
  tfn.set_transfer_function(c, o, r, nullptr);
  if (macrocell.allocated()) {
    macrocell.update_max_opacity(tfn.tfn, nullptr);
  }
}

void SimpleVolume::set_data_transform(affine3f transform)
{
  sampler->set_transform(transform);
}

INSTANT_VNR_NAMESPACE_END
