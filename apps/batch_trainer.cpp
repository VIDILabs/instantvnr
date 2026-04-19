//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //

#if defined(_WIN32)
#include <windows.h>
#endif

#include "cmdline.h"

#include <api.h>

#include <vidi_progress_bar.h>
#include <vidi_highperformance_timer.h>
#include <vidi_logger.h>

using Timer  = vidi::details::HighPerformanceTimer;
using Logger = vidi::CsvLogger;

#define BAD_LOSS 0.9f

static float estpsnr(float l1) { return (float)(-10. * log10(l1)); }

struct CmdArgs : CmdArgsBase {
public:
  args::ArgumentParser parser;
  args::HelpFlag help;

  args::ValueFlag<std::string> m_volume;
  args::ValueFlag<std::string> m_config;
  std::string volume() { return (m_volume) ? args::get(m_volume) : "network.json"; }
  std::string config() { return (m_config) ? args::get(m_config) : "network.json"; }

  args::ValueFlag<std::string> m_resume;
  std::string resume() { return (m_resume) ? args::get(m_resume) : std::string(); }

  args::ValueFlag<int> m_max_num_steps;
  int max_num_steps() { return (m_max_num_steps) ? args::get(m_max_num_steps) : 1000; }

  args::ValueFlag<std::string> m_training_mode;
  std::string training_mode() { return (m_training_mode) ? args::get(m_training_mode) : "GPU"; }

  args::ValueFlag<std::string> m_report;
  std::string report_filename() { return (m_report) ? args::get(m_report) : "none"; }

  args::Flag quiet;
  args::Flag train_macrocell;

public:
  CmdArgs(const char* title, int argc, char** argv)
    : parser(title)
    , help(parser, "help", "display the help menu", {'h', "help"})
    , m_volume(parser, "filename", "the ground truth volume", {"volume"})
    , m_config(parser, "filename", "the neural network model configuration", {"network"})
    , m_resume(parser, "filename", "the pre-trained neural network", {"resume"})
    , m_report(parser, "filename", "creating a trainning log file", {"report"})
    , m_max_num_steps(parser, "int", "maximum number of training steps", {"max-num-steps"})
    , m_training_mode(parser, "string", "the data sampling mode", { "training-mode", "mode" })
    , quiet(parser, "flag", "quiet mode", {"quiet"})
    , train_macrocell(parser, "flag", "train the macrocell grid at the same time", {"train-macrocell"})
  {
    exec(parser, argc, argv);
  }
};

namespace vidi {
enum VoxelType {
  VOXEL_UINT8   = vnr::VALUE_TYPE_UINT8,
  VOXEL_INT8    = vnr::VALUE_TYPE_INT8,
  VOXEL_UINT16  = vnr::VALUE_TYPE_UINT16,
  VOXEL_INT16   = vnr::VALUE_TYPE_INT16,
  VOXEL_UINT32  = vnr::VALUE_TYPE_UINT32,
  VOXEL_INT32   = vnr::VALUE_TYPE_INT32,
  VOXEL_FLOAT   = vnr::VALUE_TYPE_FLOAT,
  VOXEL_FLOAT2  = vnr::VALUE_TYPE_FLOAT2,
  VOXEL_FLOAT3  = vnr::VALUE_TYPE_FLOAT3,
  VOXEL_FLOAT4  = vnr::VALUE_TYPE_FLOAT4,
  VOXEL_DOUBLE  = vnr::VALUE_TYPE_DOUBLE,
  // multi-channel double types: vnr has no counterpart; use sentinel values
  VOXEL_DOUBLE2 = 501,
  VOXEL_DOUBLE3 = 502,
  VOXEL_DOUBLE4 = 503,
};
} // namespace vidi
#define VIDI_VOLUME_EXTERNAL_TYPE_ENUM
#include <vidi_volume_reader.h>

struct DataDesc {
    int dimx = -1;
    int dimy = -1;
    int dimz = -1;
    const char * dtype;

    int numfields = 1;

    float min = +1;
    float max = -1;

    bool enable_clipping = false;
    float clipbox[6] = { 0 };

    bool enable_scaling = false;
    float scaling[3] = { 1, 1, 1 };

    // variant 1 (flattened ND array)
    std::vector<void*> fields;
    void* fields_flatten = nullptr;

    // variant 2
    void * callback_context = nullptr;
    void (*callback)(void* ctx, void*, void*, unsigned long long) = nullptr;
};

struct VolumeDesc_Structured {
  DataDesc shape;
  const char * filename;
  unsigned long long offset;
  bool is_big_endian;

  void* dst;
};

void dvnrLoadData(VolumeDesc_Structured& desc)
{   
  void* &dst = desc.dst;

  vidi::StructuredRegularVolumeDesc volume_desc;
  volume_desc.dims.x = desc.shape.dimx;
  volume_desc.dims.y = desc.shape.dimy;
  volume_desc.dims.z = desc.shape.dimz;
  volume_desc.type = (vidi::VoxelType)vnr::VALUE_TYPE_FLOAT;
  volume_desc.offset = desc.offset;
  volume_desc.is_big_endian = desc.is_big_endian;
  vidi::read_volume_structured_regular(desc.filename, volume_desc, dst);
}

/*! main entry point to this example - initially optix, print hello
  world, then exit */
extern "C" int
main(int ac, char** av)
{
  CmdArgs args("Commandline Trainer", ac, av);
  int steps = args.max_num_steps();

  Timer timer;
  Logger logger;
  ProgressBar bar("[train]");

  vnrJson model = vnrCreateJsonText(args.config());

  // VolumeDesc_Structured desc;
  // {
  //   desc.shape.dimx = 1152;
  //   desc.shape.dimy = 320;
  //   desc.shape.dimz = 853;
  //   desc.shape.dtype = "float32";
  //   desc.offset = 0;
  //   desc.filename = "data/datasets/1atm.heatrelease.3x.1152.320.853f32.bin";
  //   desc.is_big_endian = false;
  //   desc.shape.min = -3290981376;
  //   desc.shape.max = 0;
  // }
  //
  // const size_t size = (size_t)desc.shape.dimx*(size_t)desc.shape.dimy*(size_t)desc.shape.dimz;
  // std::shared_ptr<char[]> buffer(new char[size * sizeof(float)]);
  // desc.dst = (void*)buffer.get();
  // dvnrLoadData(desc);
  //
  // vnrVolume simple_volume = vnrCreateSimpleVolume(desc.dst, 
  //   vnr::vec3i(desc.shape.dimx, desc.shape.dimy, desc.shape.dimz), "float32", 
  //   vnr::range1f(desc.shape.min, desc.shape.max),
  //   args.training_mode()
  // );

  vnrVolume simple_volume = vnrCreateSimpleVolume(args.volume(), args.training_mode());
  vnrVolume neural_volume;

restart:
  neural_volume = vnrCreateNeuralVolume(model, simple_volume, args.train_macrocell);

  if (!args.resume().empty()) {
    vnrJson params = vnrCreateJsonBinary(args.resume());
    vnrNeuralVolumeSetParams(neural_volume, params);
  }

  logger.initialize({"step", "loss"}, args.report_filename());

  for (int i = 0; i < steps; i += 10) {
    timer.start();

    vnrNeuralVolumeTrain(neural_volume, 10, true);

    timer.stop();
  
    logger.log_entry<double>({
      (double)vnrNeuralVolumeGetTrainingStep(neural_volume),
      (double)vnrNeuralVolumeGetTrainingLoss(neural_volume),
    });
  
    static char str[32];
    sprintf(str, "LOSS %f", (double)vnrNeuralVolumeGetTrainingLoss(neural_volume));

    if (!args.quiet) bar.update((float)i / steps, std::string(str));
  
    if (i >= 5000 && vnrNeuralVolumeGetTrainingLoss(neural_volume) > /*bad loss = */0.9) {
      std::cout << "bad setup, ... restart" << std::endl;
      logger.close();
      goto restart;
    }
  }

  if (!args.quiet) bar.finalize();

  const auto totaltime = timer.milliseconds();
  const auto psnr = vnrNeuralVolumeGetPSNR(neural_volume, args.report_filename().empty());
  const auto ssim = vnrNeuralVolumeGetSSIM(neural_volume, args.report_filename().empty());

  std::cout << "Summary" << std::endl;
  std::cout << "  STEP="<< vnrNeuralVolumeGetTrainingStep(neural_volume) << std::endl;
  std::cout << "  LOSS="<< vnrNeuralVolumeGetTrainingLoss(neural_volume) << std::endl;
  std::cout << "  TIME="<< totaltime / 1000.0 << "s"<< std::endl;
  std::cout << "  PSNR="<< psnr << std::endl;
  std::cout << "  SSIM="<< ssim << std::endl;

  // vnrNeuralVolumeSerializeParams(neural_volume, "params.json");

  vnrJson output;
  vnrNeuralVolumeSerializeParams(neural_volume, output);
  vnrSaveJsonBinary(output, "params.json");

  simple_volume.reset();
  neural_volume.reset();

  // vnrFreeTemporaryGPUMemory();
  vnrMemoryQueryPrint("[vnr]"); // Optional

  return 0;
}
