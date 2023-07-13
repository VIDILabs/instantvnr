//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //

#include "api_internal.h"

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

using namespace vnr;

void vnrLoadJsonText(vnrJson& output, std::string filename)
{
  std::ifstream file(filename);
  output = vnr::json::parse(file, nullptr, true, true);
}

void vnrLoadJsonBinary(vnrJson& output, std::string filename)
{
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(size);
  if (file.read(buffer.data(), size)) {
    output = vnr::json::from_bson(buffer);
  }
}

void vnrSaveJsonText(const vnrJson& root, std::string filename)
{
  std::ofstream ofs(filename, std::ios::out);
  ofs << std::setw(4) << root << std::endl;
  ofs.close();
}

void vnrSaveJsonBinary(const vnrJson& root, std::string filename)
{
  const auto broot = json::to_bson(root);
  std::ofstream ofs(filename, std::ios::binary | std::ios::out);
  ofs.write((char*)broot.data(), broot.size());
  ofs.close();
}


vnrJson vnrCreateJsonText(std::string filename)
{
  vnr::json output;
  vnrLoadJsonText(output, filename);
  return output;
}

vnrJson vnrCreateJsonBinary(std::string filename)
{
  vnr::json output;
  vnrLoadJsonBinary(output, filename);
  return output;
}


// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

vnrCamera vnrCreateCamera()
{
  return std::make_shared<Camera>();
}

vnrCamera vnrCreateCamera(const vnrJson& scene)
{
  auto cam = std::make_shared<Camera>();
  if (scene.is_string()) {
    create_json_camera(scene.get<std::string>(), *cam);
  }
  else {
    create_json_camera_stringify(scene, *cam);
  }
  return cam;
}

void vnrCameraSet(vnrCamera self, vnr::vec3f from, vnr::vec3f at, vnr::vec3f up)
{
  *self = { 
    /*from*/ from,
    /* at */ at,
    /* up */ up 
  };
}

void vnrCameraSet(vnrCamera self, const vnrJson& scene)
{
  // auto cam = std::make_shared<Camera>();
  if (scene.is_string()) {
    create_json_camera(scene.get<std::string>(), *self);
  }
  else {
    create_json_camera_stringify(scene, *self);
  }
}

vnr::vec3f vnrCameraGetPosition(vnrCamera self)
{
  return self->from;
}

vnr::vec3f vnrCameraGetFocus(vnrCamera self)
{
  return self->at;
}

vnr::vec3f vnrCameraGetUpVec(vnrCamera self)
{
  return self->up;
}

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

inline std::shared_ptr<SimpleVolumeContext> 
castSimpleVolume(vnrVolume self) 
{
  if (self->isNetwork()) {
    throw std::runtime_error("expecting a simple volume");
  }
  return std::dynamic_pointer_cast<SimpleVolumeContext>(self);
}

inline std::shared_ptr<NeuralVolumeContext> 
castNeuralVolume(vnrVolume self) 
{
  if (!self->isNetwork()) {
    throw std::runtime_error("expecting a neural volume");
  }
  return std::dynamic_pointer_cast<NeuralVolumeContext>(self);
}

// simple volume

vnrVolume vnrCreateSimpleVolume(const vnrJson& scene, std::string sampling_mode, bool save_loaded_volume)
{
  auto ret = std::make_shared<SimpleVolumeContext>();
  if (scene.is_string()) {
    create_json_volume(scene.get<std::string>(), ret->desc);
  }
  else {
    create_json_volume_stringify(scene, ret->desc);
  }

  ret->source.load(ret->desc, sampling_mode, save_loaded_volume);
  ret->clipbox = box3f(vec3f(0), vec3f(1));
  return ret;
}

void vnrSimpleVolumeSetCurrentTimeStep(vnrVolume self, int time)
{
  auto sv = castSimpleVolume(self);
  sv->source.set_current_timestep(time);
}

int vnrSimpleVolumeGetNumberOfTimeSteps(vnrVolume self)
{
  auto sv = castSimpleVolume(self);
  return sv->source.get_num_timesteps();
}

// neural volume

vnrVolume vnrCreateNeuralVolume(const json& config, vnrVolume groundtruth, bool online_macrocell_construction)
{
  auto& source = castSimpleVolume(groundtruth)->source;

  auto ret = std::make_shared<NeuralVolumeContext>();
  ret->desc = groundtruth->desc;
  if (config.is_string()) {
    ret->neural.set_network(ret->desc.dims, config.get<std::string>(), &source, !online_macrocell_construction);
  }
  else {
    ret->neural.set_network_from_json(ret->desc.dims, config, &source, !online_macrocell_construction);
  }
  ret->clipbox = box3f(vec3f(0), vec3f(1));
  return ret;
}

vnrVolume vnrCreateNeuralVolume(const json& config, vnr::vec3i dims)
{
  auto ret = std::make_shared<NeuralVolumeContext>();
  ret->desc.dims = dims;
  ret->desc.type = vnr::VALUE_TYPE_FLOAT;
  ret->desc.range = range1f(0, 1);
  if (config.is_string()) {
    ret->neural.set_network(ret->desc.dims, config.get<std::string>(), nullptr, false);
  }
  else {
    ret->neural.set_network_from_json(ret->desc.dims, config, nullptr, false);
  }
  ret->clipbox = box3f(vec3f(0), vec3f(1));
  return ret;
}

vnrVolume vnrCreateNeuralVolume(const json& params)
{
  vec3i dims;
  if (params.contains("volume")) {
    dims.x = params["volume"]["dims"]["x"].get<int>();
    dims.y = params["volume"]["dims"]["y"].get<int>();
    dims.z = params["volume"]["dims"]["z"].get<int>();
  }
  else {
    throw std::runtime_error("expecting a model config with volume dims tag");
  }
  auto ret = vnrCreateNeuralVolume(params["model"], dims);
  vnrNeuralVolumeSetParams(ret, params);
  return ret;
}

void vnrNeuralVolumeTrain(vnrVolume self, int steps, bool fast_mode)
{
  auto nv = castNeuralVolume(self);
  nv->neural.train(steps, fast_mode);
}

void vnrNeuralVolumeDecodeProgressive(vnrVolume self)
{
  auto nv = castNeuralVolume(self);
  nv->neural.infer();
}

void vnrNeuralVolumeDecodeInference(vnrVolume self, std::string filename)
{
  auto nv = castNeuralVolume(self);
  nv->neural.save_inference_volume(filename, nv->neural.get_data_dims());
}

void vnrNeuralVolumeDecodeReference(vnrVolume self, std::string filename)
{
  auto nv = castNeuralVolume(self);
  nv->neural.save_reference_volume(filename, nv->neural.get_data_dims());
}

void vnrNeuralVolumeSerializeParams(vnrVolume self, std::string filename)
{
  auto nv = castNeuralVolume(self);
  nv->neural.save_params(filename);
}

void vnrNeuralVolumeSerializeParams(vnrVolume self, vnrJson& params)
{
  auto nv = castNeuralVolume(self);
  nv->neural.save_params_to_json(params);
}

void vnrNeuralVolumeSetModel(vnrVolume self, const vnrJson& config)
{
  auto nv = castNeuralVolume(self);
  if (config.is_string()) {
    nv->neural.set_network(config.get<std::string>());
  }
  else {
    nv->neural.set_network_from_json(config);
  }
}

void vnrNeuralVolumeSetParams(vnrVolume self, const vnr::json& params)
{
  auto nv = castNeuralVolume(self);
  if (params.is_string()) {
    nv->neural.load_params(params.get<std::string>());
  }
  else {
    nv->neural.load_params_from_json(params);
  }
}

double vnrNeuralVolumeGetPSNR(vnrVolume self, bool verbose)
{
  auto nv = castNeuralVolume(self);
  return nv->neural.get_psnr(self->desc.dims, !verbose);
}

double vnrNeuralVolumeGetSSIM(vnrVolume self, bool verbose)
{
  auto nv = castNeuralVolume(self);
  return nv->neural.get_mssim(self->desc.dims, !verbose);
}

double vnrNeuralVolumeGetTestingLoss(vnrVolume self)
{
  auto nv = castNeuralVolume(self);
  float loss;
  nv->neural.test(&loss);
  return loss;
}

double vnrNeuralVolumeGetTrainingLoss(vnrVolume self)
{
  auto nv = castNeuralVolume(self);
  nv->neural.statistics(nv->stats);
  return nv->stats.loss;
}

int vnrNeuralVolumeGetTrainingStep(vnrVolume self)
{
  auto nv = castNeuralVolume(self);
  nv->neural.statistics(nv->stats);
  return (int)nv->stats.step;
}

int vnrNeuralVolumeGetNumberOfBlobs(vnrVolume self)
{
  auto nv = castNeuralVolume(self);
  return nv->neural.get_num_blobs();
}

// general

void vnrVolumeSetClippingBox(vnrVolume self, vnr::vec3f lower, vnr::vec3f upper)
{
  vnr::affine3f transform;
  if (self->isNetwork()) {
    transform = std::dynamic_pointer_cast<NeuralVolumeContext>(self)->neural.get_data_transform();
  }
  else {
    transform = std::dynamic_pointer_cast<SimpleVolumeContext>(self)->source.get_data_transform();
  }

  lower -= vec3f(self->desc.dims)/2;
  upper -= vec3f(self->desc.dims)/2;
  lower = gdt::xfmPoint(transform.inverse(), lower);
  upper = gdt::xfmPoint(transform.inverse(), upper);
  self->clipbox.lower = lower;
  self->clipbox.upper = upper;
}

void vnrVolumeSetScaling(vnrVolume self, vnr::vec3f scale)
{
  VolumeObject* v = nullptr;
  if (self->isNetwork()) {
    v = &(std::dynamic_pointer_cast<NeuralVolumeContext>(self)->neural);
  }
  else {
    v = &(std::dynamic_pointer_cast<SimpleVolumeContext>(self)->source);
  }
  vnr::affine3f transform = vnr::affine3f::scale(scale) * v->get_data_transform();
  v->set_data_transform(transform);
}

vnr::range1f vnrVolumeGetValueRange(vnrVolume self)
{
  if (self->isNetwork()) {
    return std::dynamic_pointer_cast<NeuralVolumeContext>(self)->neural.get_data_value_range();
  }
  else {
    return std::dynamic_pointer_cast<SimpleVolumeContext>(self)->source.get_data_value_range();
  }
}

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

vnrTransferFunction vnrCreateTransferFunction()
{
  return std::make_shared<TransferFunction>();
}

vnrTransferFunction vnrCreateTransferFunction(const vnr::json& scene)
{
  auto tfn = std::make_shared<TransferFunction>();
  if (scene.is_string()) {
    create_json_tfn(scene.get<std::string>(), *tfn);
  }
  else {
    create_json_tfn_stringify(scene, *tfn);
  }
  return tfn;
}

void vnrTransferFunctionSetColor(vnrTransferFunction tfn, const std::vector<vnr::vec3f>& colors)
{
  tfn->color = colors;
}

void vnrTransferFunctionSetAlpha(vnrTransferFunction tfn, const std::vector<vnr::vec2f>& alphas)
{
  tfn->alpha = alphas;
}

void vnrTransferFunctionSetValueRange(vnrTransferFunction tfn, vnr::range1f range)
{
  tfn->range = range;
}

const std::vector<vnr::vec3f>& vnrTransferFunctionGetColor(vnrTransferFunction tfn)
{
  return tfn->color;
}

const std::vector<vnr::vec2f>& vnrTransferFunctionGetAlpha(vnrTransferFunction tfn)
{
  return tfn->alpha;
}

const vnr::range1f& vnrTransferFunctionGetValueRange(vnrTransferFunction tfn)
{
  return tfn->range;
}


// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

vnrRenderer vnrCreateRenderer(vnrVolume v)
{
  auto self = std::make_shared<RendererContext>();
  self->volume = v;

  auto& renderer = self->renderer;
  if (self->volume->isNetwork()) {
    auto& neural = std::dynamic_pointer_cast<NeuralVolumeContext>(self->volume)->neural;

    // std::cout << "INF MC " <<  neural.get_macrocell_value_range() << std::endl;
    renderer.set_scene(neural.texture(), 
                       neural.get_data_type(), 
                       neural.get_data_dims(), 
                       neural.get_data_value_range(), 
                       neural.get_data_transform(), 
                       neural.get_macrocell_dims(), 
                       neural.get_macrocell_spacings(), 
                       neural.get_macrocell_value_range(), 
                       neural.get_macrocell_max_opacity(), 
                       &neural);
  }
  else {
    auto& source = std::dynamic_pointer_cast<SimpleVolumeContext>(self->volume)->source;

    // std::cout << "REF MC " <<  source.get_macrocell_value_range() << std::endl;
    renderer.set_scene(source.texture(), 
                       source.get_data_type(), 
                       source.get_data_dims(), 
                       source.get_data_value_range(), 
                       source.get_data_transform(), 
                       source.get_macrocell_dims(), 
                       source.get_macrocell_spacings(), 
                       source.get_macrocell_value_range(), 
                       source.get_macrocell_max_opacity());
  }

  renderer.set_scene_clipbox(self->volume->clipbox);
  renderer.set_rendering_mode(5);
  renderer.init();
  return self;
}

void vnrRendererSetMode(vnrRenderer self, int mode)
{
  self->renderer.set_rendering_mode(mode);
}

void vnrRendererSetDenoiser(vnrRenderer self, bool flag)
{
  self->renderer.set_denoiser(flag);
}

void vnrRendererSetVolumeSamplingRate(vnrRenderer self, float rate)
{
  self->renderer.set_volume_sampling_rate(rate);
}

void vnrRendererSetVolumeDensityScale(vnrRenderer self, float value)
{
  self->renderer.set_volume_density_scale(value);
}

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

void vnrRendererSetTransferFunction(vnrRenderer self, vnrTransferFunction _tfn)
{
  auto& tfn = *_tfn;
  if (self->volume->isNetwork()) {
    auto nv = std::dynamic_pointer_cast<NeuralVolumeContext>(self->volume);
    nv->neural.set_transfer_function(tfn.color, tfn.alpha, tfn.range);
  }
  else {
    auto sv = std::dynamic_pointer_cast<SimpleVolumeContext>(self->volume);
    sv->source.set_transfer_function(tfn.color, tfn.alpha, tfn.range);
  }

  self->renderer.set_transfer_function(tfn.color, tfn.alpha, tfn.range);
}

void vnrRendererSetCamera(vnrRenderer self, vnrCamera cam)
{
  self->renderer.set_camera(*cam);
}

void vnrRendererSetFramebufferSize(vnrRenderer self, vec2i fbsize)
{
  self->renderer.resize(fbsize);
}

vnr::vec4f *vnrRendererMapFrame(vnrRenderer self)
{
  vec4f *pixels = nullptr;
  self->renderer.mapframe(&pixels);
  return pixels;
}

void vnrRendererResetAccumulation(vnrRenderer self)
{
  self->renderer.reset_frame();
}

void vnrRender(vnrRenderer self)
{
  self->renderer.render();
}


// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

void vnrMemoryQuery(size_t* used_by_renderer, size_t* used_by_tcnn)
{
  *used_by_renderer = util::total_n_bytes_allocated();
  *used_by_tcnn = NeuralVolume::total_n_bytes_allocated_by_tcnn();
}

void vnrMemoryQueryPrint(const char* str)
{
  size_t used_by_renderer;
  size_t used_by_tcnn;
  unsigned long long used_total;
  vnrMemoryQuery(&used_by_renderer, &used_by_tcnn);
  util::getUsedGPUMemory(&used_total);

  printf("%s: total used %s, engine %s, tcnn %s, unknown %s\n", str,
         util::prettyBytes(used_total).c_str(),
         util::prettyBytes(used_by_renderer).c_str(),
         util::prettyBytes(used_by_tcnn).c_str(),
         util::prettyBytes(used_total - used_by_renderer - used_by_tcnn).c_str()
  );
}

void vnrFreeTemporaryGPUMemory()
{
  NeuralVolume::free_temporary_gpu_memory_by_tcnn();
}
