//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2023 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //
#if defined(_WIN32)
#include <windows.h>
#endif

#include "cmdline.h"

#include <api.h>
#include <cuda/cuda_buffer.h>
// #define STB_IMAGE_IMPLEMENTATION
// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #define STBI_MSC_SECURE_CRT
// #include <stbi/stb_image.h>
// #include <stbi/stb_image_write.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <mutex>
#include <thread>

#include <json/json.hpp>

using json = nlohmann::json;
using namespace vnr::math;


struct CmdArgs : CmdArgsBase {
public:
  args::ArgumentParser parser;
  args::HelpFlag help;

  args::Positional<std::string> m_volume;
  std::string volume() { return args::get(m_volume); }

  args::ValueFlag<std::string> m_groundtruth;
  bool has_groundtruth() { return (m_groundtruth); }
  std::string groundtruth() { return args::get(m_groundtruth); }

  args::ValueFlag<vec3f, args_impl::Vec3fReader> m_dims;
  bool has_dims() { return (m_dims); }
  vec3i dims() { return (m_dims) ? (vec3i)args::get(m_dims) : vec3i(0, 0, 0); }

  args::Flag correct;

public:
  CmdArgs(const char* title, int argc, char** argv)
    : parser(title)
    , help(parser, "help", "display the help menu", {'h', "help"})
    , m_volume(parser, "filename", "the neural volume", {"volume"})
    , m_groundtruth(parser, "filename", "the ground truth volume", {"groundtruth"})
    , m_dims(parser, "vec3i", "volume dimension", {"dims"})
    , correct(parser, "flag", "correct model", {"correct"})
  {
    exec(parser, argc, argv);
  }
};

int main(int ac, char** av) 
{

  // -------------------------------------------------------
  // initialize command line arguments
  // -------------------------------------------------------
  CmdArgs args("Model Viewer", ac, av);

  std::string volume = args.volume();

  vnrJson root;
  vnrLoadJsonBinary(root, volume);

  if (root.contains("volume")) {
    const vec3i dims = vec3i(root["volume"]["dims"]["x"].get<int>(),
                             root["volume"]["dims"]["y"].get<int>(),
                             root["volume"]["dims"]["z"].get<int>());
    std::cout << "[info] volume dims: " << dims;
  }
  else {
    std::cout << "[info] this file does not contain dimension data." << std::endl;
    if (args.correct && args.has_dims()) {
      root["volume"]["dims"]["x"] = args.dims().x;
      root["volume"]["dims"]["y"] = args.dims().y;
      root["volume"]["dims"]["z"] = args.dims().z;
    }
  }

  if (root.contains("macrocell")) {
    const bool use_reference_macrocell = root["macrocell"]["groundtruth"].get<bool>();
    const vec3i mcdims = vec3i(root["macrocell"]["dims"]["x"].get<int>(),
                               root["macrocell"]["dims"]["y"].get<int>(),
                               root["macrocell"]["dims"]["z"].get<int>());
    const vec3f mcspac = vec3f(root["macrocell"]["spacings"]["x"].get<float>(),
                               root["macrocell"]["spacings"]["y"].get<float>(),
                               root["macrocell"]["spacings"]["z"].get<float>());
    const json::binary_t mcdata = root["macrocell"]["data"];

    std::cout << "[info] use GT macrocell = " << use_reference_macrocell << std::endl;
    std::cout << "[info] macrocell dims = "    << mcdims << std::endl;
    std::cout << "[info] macrocell spacing = " << mcspac << std::endl;
    std::cout << "[info] macrocell data = "    << util::prettyBytes(mcdata.size()) << std::endl;
  }
  else {
    std::cout << "[info] this file does not contain macrocell data." << std::endl;
  }

  if (root.contains("model")) {
    std::cout << "[info] model: " << root["model"].dump(2) << std::endl;
  }
  else {
    std::cout << "[info] this file does not contain model information." << std::endl;
  }

  if (root.contains("parameters")) {
    json::binary_t params = root["parameters"]["params_binary"];
    std::cout << "[info] params = " << util::prettyBytes(params.size()) << std::endl;
  }
  else {
    std::cout << "[info] this file does not contain model weights?!" << std::endl;
  }

  if (args.correct) {
    std::cout << "Corrected model '" << volume << "' and saved it as 'params-corrected.json'." << std::endl;
    vnrSaveJsonBinary(root, "params-corrected.json");
  }

  if (args.has_groundtruth()) {
    auto groundtruth = vnrCreateSimpleVolume(args.groundtruth(), "GPU", false);

    auto volume = vnrCreateNeuralVolume(root["model"], groundtruth, true);
    vnrNeuralVolumeSetParams(volume, root);

    const auto psnr = vnrNeuralVolumeGetPSNR(volume, false);
    const auto ssim = vnrNeuralVolumeGetSSIM(volume, false);
    std::cout << "Summary" << std::endl;
    std::cout << "  PSNR="<< psnr << std::endl;
    std::cout << "  SSIM="<< ssim << std::endl;
  }

  return 0;
}