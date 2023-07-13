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
#include <core/marching_cube.cuh>

#include <vidi_highperformance_timer.h>

using Timer  = vidi::details::HighPerformanceTimer;

struct CmdArgs : CmdArgsBase {
public:
  args::ArgumentParser parser;
  args::HelpFlag help;
  args::Group group_volume;
  args::Group group_required;

  args::ValueFlag<std::string> m_simple_volume;
  args::ValueFlag<std::string> m_neural_volume;
  bool has_simple_volume() { return m_simple_volume; }
  bool has_neural_volume() { return m_neural_volume; }
  std::string volume() { return (m_simple_volume) ? args::get(m_simple_volume) : args::get(m_neural_volume); }

  args::ValueFlag<float> m_isovalue;
  float isovalue() { return args::get(m_isovalue); }

public:
  CmdArgs(const char* title, int argc, char** argv)
    : parser(title)
    , help(parser, "help", "display the help menu", {'h', "help"})
    , group_volume(parser, "Must Provide One of the Following Arguments:", args::Group::Validators::Xor)
    , group_required(parser, "Required Arguments:", args::Group::Validators::All)
    , m_simple_volume(group_volume, "filename", "the simple volume to render", {"simple-volume"})
    , m_neural_volume(group_volume, "filename", "the neural volume to render", {"neural-volume"})
    , m_isovalue(group_required, "float", "iso-value", {"iso", "isovalue"})
  {
    exec(parser, argc, argv);
  }
};

/*! main entry point to this example - initially optix, print hello
  world, then exit */
extern "C" int
main(int ac, char** av)
{
  // -------------------------------------------------------
  // initialize command line arguments
  // -------------------------------------------------------
  CmdArgs args("Commandline Volume Renderer", ac, av);

  vnrVolume volume;
  if (args.has_simple_volume()) {
    volume = vnrCreateSimpleVolume(args.volume(), "GPU", false);
  }
  else {
    vnrJson params;
    vnrLoadJsonBinary(params, args.volume());
    volume = vnrCreateNeuralVolume(params);
  }

  vnr::vec3f* data;
  size_t size;
  vnrMarchingCube(volume, args.isovalue(), &data, &size, false);
  vnrSaveTriangles("isosurface.obj", data, size);
  delete[] data;

  return 0;
}
