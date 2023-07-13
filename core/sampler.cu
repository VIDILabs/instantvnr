#include "sampler.h"

INSTANT_VNR_NAMESPACE_BEGIN

void
SimpleVolume::load(const MultiVolume& descriptor, std::string sampling_mode, bool save_volume)
{
  desc = descriptor;
  mode = sampling_mode;
  sampler.load(desc, sampling_mode, save_volume);
  tex = sampler.texture();
  if (tex) {
    macrocell.set_shape(desc.dims);
    macrocell.allocate();
    macrocell.compute_everything(tex);
  }
}

void 
SimpleVolume::set_current_timestep(int index) 
{ 
  sampler.set_current_volume_index(index); 
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
  sampler.set_transform(transform);
}

INSTANT_VNR_NAMESPACE_END
