//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2023 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //
//. ======================================================================== //
//. Copyright 2018-2019 Ingo Wald                                            //
//.                                                                          //
//. Licensed under the Apache License, Version 2.0 (the "License");          //
//. you may not use this file except in compliance with the License.         //
//. You may obtain a copy of the License at                                  //
//.                                                                          //
//.     http://www.apache.org/licenses/LICENSE-2.0                           //
//.                                                                          //
//. Unless required by applicable law or agreed to in writing, software      //
//. distributed under the License is distributed on an "AS IS" BASIS,        //
//. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
//. See the License for the specific language governing permissions and      //
//. limitations under the License.                                           //
//. ======================================================================== //

#if defined(_WIN32)
#include <windows.h>
#endif

// it is necessary to include glad before glfw
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// our helper library for window handling
#include <glfwapp/GLFWApp.h>

#include <imgui.h>
#include <implot.h>

#include <api.h>

#include "cmdline.h"

#include <cuda/cuda_buffer.h>
#ifdef NDEBUG
#define TRACE_CUDA ((void)0)
#else
#define TRACE_CUDA CUDA_SYNC_CHECK()
#endif
#ifdef NDEBUG
#define ASSERT_THROW(X, MSG) ((void)0)
#else
#define ASSERT_THROW(X, MSG) { if (!(X)) throw std::runtime_error(MSG); }
#endif

#include <vidi_async_loop.h>
#include <vidi_transactional_value.h>
#include <vidi_fps_counter.h>
#include <vidi_highperformance_timer.h>
#include <vidi_logger.h>

namespace tfn {
typedef vnr::math::vec2f vec2f;
typedef vnr::math::vec2i vec2i;
typedef vnr::math::vec3f vec3f;
typedef vnr::math::vec3i vec3i;
typedef vnr::math::vec4f vec4f;
typedef vnr::math::vec4i vec4i;
} // namespace tfn
#define TFN_MODULE_EXTERNAL_VECTOR_TYPES
#include <tfn/widget.h>
using tfn::TransferFunctionWidget;

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#include <stbi/stb_image.h>
#include <stbi/stb_image_write.h>

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

#define SCENE_SCALE 1024

using namespace vnr::math;

struct Camera {
  vec3f from;
  vec3f at;
  vec3f up;
  float fovy = 60;
};

struct TransferFunction {
  std::vector<vec3f> color;
  std::vector<vec2f> alpha;
  range1f range;
};

using vidi::TransactionalValue;
using vidi::FPSCounter;
using vidi::AsyncLoop;

using Timer = vidi::StackTimer;
using TextLogger = vidi::CsvLogger;
struct LossLogger {
  std::atomic<float>  loss { 0.f  };
  std::atomic<size_t> step { 0ULL };

  std::atomic<bool> local_updated { false };
  std::vector<float> local_steps;
  std::vector<float> local_loss_history;

  const size_t size;
  std::mutex mutex;

  LossLogger(size_t size) : size(size) 
  {
    local_steps.reserve(size);
    local_loss_history.reserve(size);
  }

  void reset()
  {
    step = 0;
    loss = 0.f;
    
    local_steps.clear();
    local_steps.reserve(size);

    local_loss_history.clear();
    local_loss_history.reserve(size);
  }

  void update(float stats_loss, float stats_step) 
  {
    std::lock_guard<std::mutex> lock{ mutex };

    loss = stats_loss;
    step = stats_step;

    if (local_steps.size() > size) {
      local_steps.erase(local_steps.begin());
      local_loss_history.erase(local_loss_history.begin());
    }

    local_steps.push_back(stats_step);
    local_loss_history.push_back(stats_loss);
    local_updated = true;
  }

  void get_local_history(std::vector<float>& x, std::vector<float>& y)
  {
    std::lock_guard<std::mutex> lock{ mutex };
    x = local_steps;
    y = local_loss_history;
    local_updated = false;
  }
};

struct Args {
public:
  using vec3f = gdt::vec3f;

  std::string volume_filename = "network.json";
  std::string config_filename = "network.json";
  std::string resume = "";
  bool online_macrocell = true;
  bool save_reference_volume = false;
  uint32_t max_num_frames = 0;
  uint32_t max_num_steps = 0;
  std::string training_mode = "GPU";
  int   rendering_mode = 0;
  float sampling_rate = 1.f;
  float density_scale = 1.f;
  bool quiet = false;
  std::string report_filename = "none";

  bool pause_training = false;
  bool pause_inf_rendering = false;
  bool pause_ref_rendering = false;
  int current_timestep = 0;

  bool force_camera = false;
  vec3f camera_from=vec3f(0.f, 0.f, -1000.f);/*! camera position - *from* where we are looking */
  vec3f camera_at=vec3f(0.f, 0.f, 0.f);/*! which point we are looking *at* */
  vec3f camera_up=vec3f(0.f, 1.f, 0.f);/*! general up-vector */

  bool summary = false;
  bool report_macrocell_quality = false;
  bool report_rendering_fps = false;

  bool fvsrn = false;

public:
  Args(int ac, char** av)
  {
    int ai = 1;
    while (ai < ac) {
      std::string arg(av[ai++]);
      if (arg == "--volume") {
        if (ac < ai + 1) throw std::runtime_error("improper --volume");
        volume_filename = std::string(av[ai++]);
      }
      else if (arg == "--network") {
        if (ac < ai + 1) throw std::runtime_error("improper --network");
        config_filename = std::string(av[ai++]);
      }
      else if (arg == "--resume") {
        if (ac < ai + 1) throw std::runtime_error("improper --resume");
        resume = std::string(av[ai++]);
      }
      else if (arg == "--groundtruth-macrocell") {
        online_macrocell = false;
      }
      else if (arg == "--save-reference-volume") {
        save_reference_volume = true;
      }
      else if (arg == "--training-mode" || arg == "--mode") {
        if (ac < ai + 1) throw std::runtime_error("improper --training-mode | mode");
        training_mode = std::string(av[ai++]);
      }
      else if (arg == "--rendering-mode") {
        if (ac < ai + 1) throw std::runtime_error("improper --rendering-mode");
        rendering_mode = std::stoi(av[ai++]);
      }
      else if (arg == "--max-frames") {
        if (ac < ai + 1) throw std::runtime_error("improper --max-frames");
        max_num_frames = std::stoi(av[ai++]);
      }
      else if (arg == "--max-steps" || arg == "--steps") {
        if (ac < ai + 1) throw std::runtime_error("improper --max-steps | steps");
        max_num_steps = std::stoi(av[ai++]);
      }
      else if (arg == "--pause-training") {
        pause_training = true;
      }
      else if (arg == "--pause-reference") {
        pause_ref_rendering = true;
      }
      else if (arg == "--pause-inference") {
        pause_inf_rendering = true;
      }
      else if (arg == "--quiet") {
        quiet = true;
      }
      else if (arg == "--summary") {
        summary = true;
      }
      else if (arg == "--report-macrocell-quality") {
        report_macrocell_quality = true;
      }
      else if (arg == "--report-rendering-fps") {
        report_rendering_fps = true;
      }
      else if (arg == "--report") {
        if (ac < ai + 1) throw std::runtime_error("improper --report");
        report_filename = std::string(av[ai++]);
      }
      else if (arg == "--camera-from") {
        if (ac < ai + 3) throw std::runtime_error("improper --camera-from");
        camera_from.x = std::stof(av[ai++]);
        camera_from.y = std::stof(av[ai++]);
        camera_from.z = std::stof(av[ai++]);
        force_camera = true;
      }
      else if (arg == "--camera-at") {
        if (ac < ai + 3) throw std::runtime_error("improper --camera-at");
        camera_at.x = std::stof(av[ai++]);
        camera_at.y = std::stof(av[ai++]);
        camera_at.z = std::stof(av[ai++]);
        force_camera = true;
      }
      else if (arg == "--camera-up") {
        if (ac < ai + 3) throw std::runtime_error("improper --camera-up");
        camera_up.x = std::stof(av[ai++]);
        camera_up.y = std::stof(av[ai++]);
        camera_up.z = std::stof(av[ai++]);
        force_camera = true;
      }
      else if (arg == "--fvsrn") {
        fvsrn = true;
      }
      else {
        throw std::runtime_error("unknown switch argument '" + arg + "'");
      }
    }

    if (force_camera) {
      std::cout << GDT_TERMINAL_GREEN << "over-writting camera position" << GDT_TERMINAL_RESET << std::endl;
      std::cout << "(C)urrent camera:" << std::endl;
      std::cout << "- from :" << camera_from << std::endl;
      std::cout << "- poi  :" << camera_at << std::endl;
      std::cout << "- upVec:" << camera_up << std::endl;
    }
  }
};

struct MainWindow : public glfwapp::GLFCameraWindow {
public:
  vec2i fb_size_fg;
  TransactionalValue<vec2i> fb_size_bg; // produced by FG, consumed by BG

  struct View {
    vec4f* pixels{ nullptr };
    vec2i size;
  };

  TransactionalValue<View> view_ref, view_inf; // produced by BG, consumed by FG
  GLuint texture_ref = 0, texture_inf = 0;

  vnrRenderer renderer_ref, renderer_inf;
  vnrVolume neural_volume, simple_volume;
  vnrTransferFunction tfn;
  vnrCamera cam;

  TransferFunctionWidget widget;

  FPSCounter fps_fg, fps_bg;
  LossLogger stats = LossLogger(500);
  TextLogger training_logger;
  AsyncLoop background_task;

  // renderer parameters
  TransactionalValue<TransferFunction> transfer_function;
  TransactionalValue<Camera> camera;
  TransactionalValue<float> volume_sampling_rate;
  TransactionalValue<float> volume_density_scale;
  TransactionalValue<int> rendering_mode;
  TransactionalValue<int> current_timestep;

  // control flows
  std::atomic<int>  train_steps = 1;
  std::atomic<int>  infer_steps = 1;
  std::atomic<bool> fastforward = false;
  std::atomic<bool> trigger_test = false;
  std::atomic<bool> pause_inf = false;
  std::atomic<bool> pause_ref = false;
  std::atomic<bool> pause_training = false;
  std::atomic<bool> full_decode = false;
  std::atomic<bool> network_reset = false;
  std::atomic<bool> frame_reset = false;
  std::atomic<bool> disable_frame_accum = false;
  std::atomic<bool> save_volume = false;
  std::atomic<bool> save_params = false;
  std::atomic<bool> load_params = false;
  std::atomic<bool> compute_psnr = false;
  std::atomic<bool> compute_ssim = false;
  std::atomic<bool> denoise = false;

  const Args& args;
  size_t frame_counter = 0;

public:
  MainWindow(const Args& commandline, const std::string& title, Camera camera, const float worldScale)
    : GLFCameraWindow(title, camera.from, camera.at, camera.up, worldScale, 768 * 2, 768)
    , widget(std::bind(&MainWindow::set_transfer_function, this,
                       std::placeholders::_1,
                       std::placeholders::_2,
                       std::placeholders::_3))
    , background_task(std::bind(&MainWindow::background_work, this))
    , args(commandline)
  {
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glGenTextures(1, &texture_ref);
    glGenTextures(1, &texture_inf);

    cam = vnrCreateCamera();
    vnrCameraSet(cam, camera.from, camera.at, camera.up);
    vnrCameraSet(cam, args.volume_filename); 

    simple_volume = vnrCreateSimpleVolume(args.volume_filename, args.training_mode, args.save_reference_volume);

    if (args.fvsrn) {
      vnrJson params = vnrCreateJsonBinary(args.config_filename);
      neural_volume = vnrCreateNeuralVolume(params["model"], simple_volume, true);
      vnrNeuralVolumeSetParams(neural_volume, params);
    }
    else {
      neural_volume = vnrCreateNeuralVolume(args.config_filename, simple_volume, args.online_macrocell);
    }

    if (!args.resume.empty()) {
      vnrJson params = vnrCreateJsonBinary(args.resume);
      vnrNeuralVolumeSetParams(neural_volume, params);
    }

    std::cout << "[vnr] training mode = " << args.training_mode << std::endl;
    std::cout << "[vnr] # of inference blobs = " << vnrNeuralVolumeGetNumberOfBlobs(neural_volume) << std::endl;

    renderer_inf = vnrCreateRenderer(neural_volume);
    renderer_ref = vnrCreateRenderer(simple_volume);

    vnrRendererSetCamera(renderer_inf, cam);
    vnrRendererSetCamera(renderer_ref, cam);

    tfn = vnrCreateTransferFunction(args.volume_filename);
    vnrTransferFunctionSetValueRange(tfn, range1f(0, 1));
    vnrRendererSetTransferFunction(renderer_inf, tfn);
    vnrRendererSetTransferFunction(renderer_ref, tfn);

    vnrRendererResetAccumulation(renderer_inf);
    vnrRendererResetAccumulation(renderer_ref);

    CUDA_SYNC_CHECK(); // sanity check

    // other params
    pause_training = args.pause_training;
    pause_ref = args.pause_ref_rendering;
    pause_inf = args.pause_inf_rendering;
    volume_sampling_rate = args.sampling_rate;
    volume_density_scale = args.density_scale;
    rendering_mode = args.rendering_mode;
    current_timestep = args.current_timestep;

    // logger
    std::vector<std::string> header = {"step", "loss", "rendering_time", "training_time", "training_percentage", "fps"};
    if (args.report_macrocell_quality) {
      header.push_back("macrocell_min_reconstruction");
      header.push_back("macrocell_max_reconstruction");
    }
    training_logger.initialize(header, args.report_filename);

    // setup camera
    if (!args.force_camera) {
      camera.from = vnrCameraGetPosition(cam);
      camera.at = vnrCameraGetFocus(cam);
      camera.up = vnrCameraGetUpVec(cam);

      cameraFrame.setOrientation(camera.from, camera.at, camera.up);
    }
  
    // setup transfer function widget
    initialize_transfer_function_widget();

    // start the background now
    background_task.start();
  }

  void initialize_transfer_function_widget()
  {
    auto& color = vnrTransferFunctionGetColor(tfn);
    auto& alpha = vnrTransferFunctionGetAlpha(tfn);
    auto& range = vnrTransferFunctionGetValueRange(tfn);

    ASSERT_THROW(vnrVolumeGetValueRange(neural_volume) == range, "expecting the same value range");
    ASSERT_THROW(vnrVolumeGetValueRange(simple_volume) == range, "expecting the same value range");

    if (!color.empty()) {
      std::vector<vec4f> color_controls;
      for (int i = 0; i < color.size(); ++i) {
        color_controls.push_back(vec4f(i / float(color.size() - 1), /* control point position */
                                       color.at(i).x, color.at(i).y, color.at(i).z));
      }
      assert(!alpha.empty());
      widget.add_tfn(color_controls, alpha, "builtin");

      transfer_function.assign([&](TransferFunction& value) {
        value.color = color;
        value.alpha = alpha;
        value.range = range;
      });
    }
    widget.set_default_value_range(range.lower, range.upper);
  }

  void set_transfer_function(const std::vector<vec3f>& c, const std::vector<vec2f>& a, const vec2f& r)
  {
    transfer_function.assign([&c, &a, &r](TransferFunction& value) {
      value.color = c;
      value.alpha = a;
      value.range.lower = r.x;
      value.range.upper = r.y;
    });
  }

  void render() override
  {
    if (cameraFrame.modified) {
      camera.assign([&](Camera& c) {
        c.from = cameraFrame.get_position();
        c.at = cameraFrame.get_poi();
        c.up = cameraFrame.get_accurate_up();
      });
      cameraFrame.modified = false;
    }
  }

  void background_work()
  {
    TRACE_CUDA;

    if (current_timestep.update()) {
      vnrSimpleVolumeSetCurrentTimeStep(simple_volume, current_timestep.get());
      vnrRendererResetAccumulation(renderer_ref);
      vnrRendererResetAccumulation(renderer_inf);
    }

    TRACE_CUDA;

    if (fastforward) {
      std::cout << "fast forwatd training ... ";
      {
        Timer timer;
        vnrNeuralVolumeTrain(neural_volume, 500, false);
      }
      {
        stats.update(
          vnrNeuralVolumeGetTrainingLoss(neural_volume),
          vnrNeuralVolumeGetTrainingStep(neural_volume)
        );
        std::cout <<   "step=" << vnrNeuralVolumeGetTrainingStep(neural_volume)
                  << "  loss=" << vnrNeuralVolumeGetTrainingLoss(neural_volume) 
                  << std::endl;
      }
      fastforward = false;
      return;
    }

    TRACE_CUDA;

    if (rendering_mode.update()) {
      vnrRendererSetMode(renderer_ref, rendering_mode.get());
      vnrRendererSetMode(renderer_inf, rendering_mode.get());
    }

    if (full_decode) {
      if (vnrRequireDecoding(rendering_mode.get())) {
        for (int i = 0; i < vnrNeuralVolumeGetNumberOfBlobs(neural_volume); ++i) { 
          vnrNeuralVolumeDecodeProgressive(neural_volume);
        }
        vnrRendererResetAccumulation(renderer_ref);
        vnrRendererResetAccumulation(renderer_inf);
      }
      full_decode = false;
    }

    if (save_volume) {
      vnrNeuralVolumeDecodeReference(neural_volume, "reference.bin");
      vnrNeuralVolumeDecodeInference(neural_volume, "inference.bin");
      save_volume = false;
    }

    if (save_params) {
      vnrNeuralVolumeSerializeParams(neural_volume, "params.json");
      save_params = false;
    }
    if (load_params) {
      vnrNeuralVolumeSetParams(neural_volume, "params.json");
      load_params = false;
    }

    if (compute_psnr) {
      const auto psnr = vnrNeuralVolumeGetPSNR(neural_volume, true);
      compute_psnr = false;
      std::cout << "[vnr] PSNR: " << psnr << std::endl;
    }
    if (compute_ssim) {
      const auto ssim = vnrNeuralVolumeGetSSIM(neural_volume, true);
      compute_ssim = false;
      std::cout << "[vnr] SSIM: " << ssim << std::endl;
    }

    if (fb_size_bg.update()) {
      vnrRendererSetFramebufferSize(renderer_ref, fb_size_bg.get());
      vnrRendererSetFramebufferSize(renderer_inf, fb_size_bg.get());
    }

    TRACE_CUDA;
  
    if (transfer_function.update()) {
      vnrTransferFunctionSetColor(tfn, transfer_function.ref().color);
      vnrTransferFunctionSetAlpha(tfn, transfer_function.ref().alpha);
      vnrTransferFunctionSetValueRange(tfn, transfer_function.ref().range);
      vnrRendererSetTransferFunction(renderer_ref, tfn);
      vnrRendererSetTransferFunction(renderer_inf, tfn);
    }

    TRACE_CUDA;

    if (camera.update()) {
      vnrCameraSet(cam, camera.ref().from, camera.ref().at, camera.ref().up);
      vnrRendererSetCamera(renderer_ref, cam);
      vnrRendererSetCamera(renderer_inf, cam);
    }

    if (volume_sampling_rate.update()) {
      vnrRendererSetVolumeSamplingRate(renderer_ref, volume_sampling_rate.get());
      vnrRendererSetVolumeSamplingRate(renderer_inf, volume_sampling_rate.get());
    }
    
    if (volume_density_scale.update()) {
      vnrRendererSetVolumeDensityScale(renderer_ref, volume_density_scale.get());
      vnrRendererSetVolumeDensityScale(renderer_inf, volume_density_scale.get());
    }

    vnrRendererSetDenoiser(renderer_ref, denoise);
    vnrRendererSetDenoiser(renderer_inf, denoise);

    TRACE_CUDA;

    if (frame_reset || disable_frame_accum) {
      vnrRendererResetAccumulation(renderer_ref);
      vnrRendererResetAccumulation(renderer_inf);
      frame_reset = false;
    }

    TRACE_CUDA;

    // reload the network with a new configuration
    if (network_reset) { 
      vnrNeuralVolumeSetModel(neural_volume, args.config_filename);
      network_reset = false;
      stats.reset();
    }

    TRACE_CUDA;

    // rendering & training
    double time_rendering = 0., time_training = 0.;

    View view; // reference & inference view
    {
      auto t0 = std::chrono::high_resolution_clock::now();

      TRACE_CUDA;

      if (!pause_ref) {
        vnrRender(renderer_ref);
        view.pixels = vnrRendererMapFrame(renderer_ref);
        view.size = fb_size_bg.get();
        view_ref = view;
      }

      TRACE_CUDA;

      if (!pause_inf) {
        vnrRender(renderer_inf);
        view.pixels = vnrRendererMapFrame(renderer_inf);
        view.size = fb_size_bg.get();
        view_inf = view;
      }

      TRACE_CUDA;

      time_rendering = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

      ++frame_counter;
    }

    TRACE_CUDA;

    if (!pause_training) 
    {
      auto t0 = std::chrono::high_resolution_clock::now();

      vnrNeuralVolumeTrain(neural_volume, train_steps, false);

      if (vnrRequireDecoding(rendering_mode.get())) {
        for (int i = 0; i < infer_steps; ++i) {
          vnrNeuralVolumeDecodeProgressive(neural_volume);
        }
      }

      time_training = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

      stats.step += train_steps;
      if (stats.step % 10 == 0) 
      {
        stats.update(
          vnrNeuralVolumeGetTrainingLoss(neural_volume),
          vnrNeuralVolumeGetTrainingStep(neural_volume)
        );

        std::vector<double> log = {
          /*step=*/(double)stats.step, 
          /*loss=*/stats.loss,
          time_rendering,
          time_training,
          /*percentage=*/100.0 * time_training / (time_rendering + time_training),
          fps_bg.fps
        };

        training_logger.log_entry<double>(log);

        if (args.max_num_steps > 0 && stats.step > args.max_num_steps) {
          std::cout << GDT_TERMINAL_GREEN << "terminating because max # of training steps is reached." 
                    << GDT_TERMINAL_RESET << std::endl;
          glfwSetWindowShouldClose(handle, GLFW_TRUE); // close window
        }
      }
    }

    if (trigger_test)
    { 
      float loss = vnrNeuralVolumeGetTestingLoss(neural_volume);
      std::cout << "[test] loss = " << loss << std::endl;

      trigger_test = false;
    }

    if (args.max_num_frames > 0 && frame_counter > args.max_num_frames) {
      std::cout << GDT_TERMINAL_GREEN << "terminating because max # of rendering frame is reached." 
                << GDT_TERMINAL_RESET << std::endl;
      glfwSetWindowShouldClose(handle, GLFW_TRUE); // close window
    }

    if (fps_bg.count() && args.report_rendering_fps) {
      std::cout << "fps = " << fps_bg.fps << std::endl;
    }
  }

  static void view_draw(vec2i size, GLuint& texture)
  {
    glBindTexture(GL_TEXTURE_2D, texture);

    glColor3f(1, 1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.f, (float)size.x, 0.f, (float)size.y, -1.f, 1.f);
    glBegin(GL_QUADS);
    {
      glTexCoord2f(0.f, 0.f);
      glVertex3f(0.f, 0.f, 0.f);
      glTexCoord2f(0.f, 1.f);
      glVertex3f(0.f, (float)size.y, 0.f);
      glTexCoord2f(1.f, 1.f);
      glVertex3f((float)size.x, (float)size.y, 0.f);
      glTexCoord2f(1.f, 0.f);
      glVertex3f((float)size.x, 0.f, 0.f);
    }
    glEnd();
  }

  static void view_update(vec2i size, GLuint& texture, vec4f* pixels)
  {
    glBindTexture(GL_TEXTURE_2D, texture);

    GLenum tex_format = GL_RGBA;
    GLenum texel_type = GL_FLOAT;
    glTexImage2D(GL_TEXTURE_2D, 0, tex_format, size.x, size.y, 0, GL_RGBA, texel_type, pixels);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  }

  void gui()
  {
    ImGui::SetNextWindowSizeConstraints(ImVec2(400, 600), ImVec2(FLT_MAX, FLT_MAX));

    if (ImGui::Begin("Control Panel", NULL)) {

      // control rendering mode
      const char* render_modes = " 0 Reference (No Shading)\0"
                                 " 1 Reference (Local Illumination)\0"
                                 " 2 Reference (Full Shadow)\0"
                                 " 3 Reference (Single Shade Heuristic)\0"
                                 " 4 Ray Marching (Decoding) \0"
                                 " 5 Ray Marching (Sample Streaming) \0"
                                 " 6 Ray Marching (In Shader)\0"
                                 " 7 Ray Marching + LI (Decoding) \0"
                                 " 8 Ray Marching + LI (Sample Streaming) \0"
                                 " 9 Ray Marching + LI (In Shader)\0"
                                 "10 Ray Marching + SSH (Decoding - Debug) \0"
                                 "11 Ray Marching + SSH (Sample Streaming) \0"
                                 "12 Ray Marching + SSH (In Shader)\0"
                                 "13 Path Tracing (Decoding - Debug) \0"
                                 "14 Path Tracing (Sample Streaming) \0"
                                 "15 Path Tracing (In Shader)\0";
      static int gui_rendering_mode = args.rendering_mode;
      if (ImGui::Combo("Render Mode", &gui_rendering_mode, render_modes, IM_ARRAYSIZE(render_modes))) {
        rendering_mode = gui_rendering_mode;
      }

      // basic training and rendering behaviors
      if (ImGui::Button("Re-Configure Network")) {
        network_reset = true;
      }
      ImGui::SameLine();
      if (ImGui::Button("Reset Frame")) {
        frame_reset = true;
      }
      ImGui::SameLine();
      if (fastforward) {
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
        ImGui::Button("Training is being Fast Forwarded ...");
        ImGui::PopStyleVar();
      }
      else {
        if (ImGui::Button("Fast Forward Training")) {
          fastforward = true;
          full_decode = true;
        }
      }

      // adjust training and inferencing speeds
      static int gui_tsteps = train_steps;
      if (ImGui::SliderInt("Training Step Size", &gui_tsteps, 1, 100)) {
        train_steps = gui_tsteps;
      }
      static int gui_isteps = infer_steps;
      if (ImGui::SliderInt("Inference Step Size", &gui_isteps, 1, 100)) {
        infer_steps = gui_isteps;
      }

      // adjust rendering qualities
      static float gui_sampling_rate = args.sampling_rate;
      if (ImGui::SliderFloat("Volume Sampling Rate", &gui_sampling_rate, 0.01f, 10.f, "%.3f")) {
        volume_sampling_rate = gui_sampling_rate;
      }
      static float gui_density_scale = args.density_scale;
      if (ImGui::SliderFloat("Volume Density Scale", &gui_density_scale, 0.01f, 10.f, "%.3f")) {
        volume_density_scale = gui_density_scale;
      }

      // turn on/off training & rendering
      static bool gui_pause_training = pause_training;
      if (ImGui::Checkbox("Pause Training", &gui_pause_training)) {
        pause_training = gui_pause_training;
      }
      ImGui::SameLine();
      static bool gui_denoise_frame = denoise;
      if (ImGui::Checkbox("Denoise", &gui_denoise_frame)) {
        denoise = gui_denoise_frame;
      }
      ImGui::SameLine();
      static bool gui_render_ref = !pause_ref;
      if (ImGui::Checkbox("Render GT", &gui_render_ref)) {
        pause_ref = !gui_render_ref;
      }
      ImGui::SameLine();
      static bool gui_render_inf = !pause_inf;
      if (ImGui::Checkbox("Render NR", &gui_render_inf)) {
        pause_inf = !gui_render_inf;
      }
      ImGui::SameLine();
      static bool gui_disable_accum = false;
      if (ImGui::Checkbox("Disable Accum", &gui_disable_accum)) {
        disable_frame_accum = gui_disable_accum;
      }

      // record rendering & training results
      if (ImGui::Button("Save Screen")) {
        static int count = 0;
        saveJPG("screenshot-" + std::to_string(count++) + ".jpg");
      }
      ImGui::SameLine();
      if (ImGui::Button("Save Volume")) {
        full_decode = true;
        save_volume = true;
      }
      ImGui::SameLine();
      if (ImGui::Button("Save Params")) {
        full_decode = true;
        save_params = true;
      }
      ImGui::SameLine();
      if (ImGui::Button("Load Params")) {
        full_decode = true;
        load_params = true;
      }
      ImGui::SameLine();
      if (ImGui::Button("Test")) {
        trigger_test = true;
      }

      // measure training results
      ImGui::Text("Metric:");
      ImGui::SameLine();
      if (ImGui::Button("PSNR")) {
        full_decode = true;
        compute_psnr = true;
      }
      ImGui::SameLine();
      if (ImGui::Button("MSSIM")) {
        full_decode = true;
        compute_ssim = true;
      }

      // time varying data
      const int n_steps = vnrSimpleVolumeGetNumberOfTimeSteps(simple_volume);
      if (n_steps > 1) {
        static int idx = 0;
        if (ImGui::SliderInt("Time Step", &idx, 0, n_steps - 1)) {
          current_timestep = idx;
        }
#if 0
        static auto t_initial = std::chrono::high_resolution_clock::now();
        if ((step-1) % 10 == 0) {
          double time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_initial).count();
          if (time > (idx+1) * 5) {
            config.current_volume = ++idx;
          }
          std::cout << "time= " << time << " step= " << step-1 << "  loss= " << loss << "  fps= " << float(fps_bg.fps) << std::endl;
        }
#endif
      }

      static std::vector<float> x0, y0;
      if (stats.local_updated) {
        stats.get_local_history(x0, y0);
      }
      if (ImPlot::BeginPlot("##Loss Plot", ImVec2(500,150), ImPlotFlags_AntiAliased | ImPlotFlags_NoFrame)) {
        ImPlot::SetupAxes("Loss History", "Loss", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupLegend(ImPlotLocation_East, ImPlotLegendFlags_Outside);
        ImPlot::PlotLine("Loss", x0.data(), y0.data(), x0.size());
        ImPlot::EndPlot();
      }

      widget.build_gui();
    }
    ImGui::End();
    widget.render(util::n_threads_linear/*=128*/);
  }
  
  void title()
  {
    int width, height;
    glfwGetFramebufferSize(handle, &width, &height);
    width /= 2;

    double xpos, ypos;
    glfwGetCursorPos(handle, &xpos, &ypos);
    ypos = height - ypos + 1;

    std::stringstream title;
    title << std::fixed << std::setprecision(3) << std::setw(5) << " fg = " << fps_fg.fps << " fps,";
    title << std::fixed << std::setprecision(3) << std::setw(5) << " bg = " << fps_bg.fps << " fps,";
    title << std::fixed << std::setprecision(3) << std::setw(5) << " loss = " << stats.loss << ", step = " << stats.step;

    if (xpos >= 0 && ypos >= 0 && xpos < width && ypos < height) {
      vec4f color;
      glReadPixels(xpos, ypos, 1, 1, GL_RGBA, GL_FLOAT, &color);

      title << " pixel_index = " << (int)(xpos + ypos * width) << " color = (" << color.x << " " << color.y << " " << color.z << " " << color.w << ")";
    }

    glfwSetWindowTitle(handle, title.str().c_str());
  }

  void draw() override
  {
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, fb_size_fg.x, fb_size_fg.y);
    view_ref.update([&](const View& view) { view_update(view.size, texture_ref, view.pixels); });
    view_draw(fb_size_fg, texture_ref);

    glViewport(fb_size_fg.x, 0, fb_size_fg.x, fb_size_fg.y);
    view_inf.update([&](const View& view) { view_update(view.size, texture_inf, view.pixels); });
    view_draw(fb_size_fg, texture_inf);

    gui();

    if (fps_fg.count())
    {
      title();
    }
  }

  void saveJPG(const std::string &fname, vec2i size, vec4f* pixels)
  {
    std::vector<char> image((uint64_t)size.x*size.y*4);
    for (uint64_t i = 0; i < (uint64_t)size.x*size.y; ++i) {
      const auto in = pixels[i];
      const uint32_t r(255.99f * clamp(in.x, 0.f, 1.f));
      const uint32_t g(255.99f * clamp(in.y, 0.f, 1.f));
      const uint32_t b(255.99f * clamp(in.z, 0.f, 1.f));
      const uint32_t a(255.99f * clamp(in.w, 0.f, 1.f));
      image[4*i+0] = r;
      image[4*i+1] = g;
      image[4*i+2] = b;
      image[4*i+3] = a;
    }
    stbi_write_jpg(fname.c_str(), size.x, size.y, 4, image.data(), 100);
  }

  void saveJPG(const std::string &filename, bool sync = true)
  {
    // sync background thread to make sure that the same number of frames are rendered
    if (sync) background_task.stop();
    if (!pause_inf)
      view_inf.access([&](const View& view) { 
        auto fname = "inf-" + filename;
        stbi_flip_vertically_on_write(1);
        saveJPG("screenshots/" + fname, view.size, view.pixels);
      });
    if (!pause_ref)
      view_ref.access([&](const View& view) { 
        auto fname = "ref-" + filename;
        stbi_flip_vertically_on_write(1);
        saveJPG("screenshots/" + fname, view.size, view.pixels);
      });
    if (sync) background_task.start();
  }

  void resize(const vec2i& new_size) override
  {
    if (new_size.long_product() == 0) return;

    fb_size_fg.x = new_size.x / 2;
    fb_size_fg.y = new_size.y;

    fb_size_bg = fb_size_fg;
  }

  void close()
  {
    background_task.stop();
    glDeleteTextures(1, &texture_ref);
    glDeleteTextures(1, &texture_inf);
  }
};

extern "C" int
main(int ac, char** av)
{
  // -------------------------------------------------------
  // initialize command line arguments
  // -------------------------------------------------------
  Args args(ac, av);
  if (args.quiet) {
    std::cout << "option --quiet is not used by the interactive renderer." << std::endl;
  }

  // -------------------------------------------------------
  // initialize camera
  // -------------------------------------------------------
  Camera camera = { /*from*/ args.camera_from,
                    /* at */ args.camera_at,
                    /* up */ args.camera_up };

  // something approximating the scale of the world, so the
  // camera knows how much to move for any given user interaction:
  const float worldScale = SCENE_SCALE;

  // -------------------------------------------------------
  // initialize opengl window
  // -------------------------------------------------------
  auto* window = new MainWindow(args, "Optix 7 Renderer", camera, worldScale);

  auto t0 = std::chrono::high_resolution_clock::now();

  window->run();
  window->close();
  
  const double total = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
  std::cout << "total time = " << total << std::endl;

  if (args.summary) {
    window->saveJPG("final.jpg", false); // do not sync
    if (!window->pause_training) {
      const auto psnr = vnrNeuralVolumeGetPSNR(window->neural_volume, false);
      const auto ssim = vnrNeuralVolumeGetSSIM(window->neural_volume, false);
      std::cout << "[vnr] PSNR: " << psnr << std::endl;
      std::cout << "[vnr] SSIM: " << ssim << std::endl;
      vnrNeuralVolumeSerializeParams(window->neural_volume, "params.json");
    }
  }

  delete window;

  vnrMemoryQueryPrint("memory");

  return 0;
}
