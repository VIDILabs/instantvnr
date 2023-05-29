//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
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

using vidi::TransactionalValue;
using vidi::FPSCounter;
using vidi::AsyncLoop;

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

struct CmdArgs : CmdArgsBase {
public:
  args::ArgumentParser parser;
  args::HelpFlag help;
  args::Group group1;
  args::Group group2;

  args::ValueFlag<std::string> m_simple_volume;
  args::ValueFlag<std::string> m_neural_volume;
  bool has_simple_volume() { return m_simple_volume; }
  bool has_neural_volume() { return m_neural_volume; }
  std::string volume() { return (m_simple_volume) ? args::get(m_simple_volume) : args::get(m_neural_volume); }

  args::ValueFlag<std::string> m_tfn;
  bool has_tfn() { return m_tfn; }
  std::string tfn() { return args::get(m_tfn); }

  args::ValueFlag<vec3f, args_impl::Vec3fReader> m_camera_from; /*! camera position - *from* where we are looking */
  args::ValueFlag<vec3f, args_impl::Vec3fReader> m_camera_up;   /*! general up-vector */
  args::ValueFlag<vec3f, args_impl::Vec3fReader> m_camera_at;   /*! which point we are looking *at* */
  vec3f camera_from() { return (m_camera_from) ? args::get(m_camera_from) : vec3f(0.f, 0.f, -1000.f); }
  vec3f camera_at()   { return (m_camera_at)   ? args::get(m_camera_at)   : vec3f(0.f, 0.f, 0.f);     }
  vec3f camera_up()   { return (m_camera_up)   ? args::get(m_camera_up)   : vec3f(0.f, 1.f, 0.f);     }

  args::ValueFlag<float> m_sampling_rate;
  float sampling_rate() { return (m_sampling_rate) ? args::get(m_sampling_rate) : 1.f; }
  
  args::ValueFlag<float> m_density_scale;
  float density_scale() { return (m_density_scale) ? args::get(m_density_scale) : 1.f; }
  
  args::ValueFlag<int> m_rendering_mode;
  float rendering_mode() { return (m_rendering_mode) ? args::get(m_rendering_mode) : 0; }

  args::Flag force_camera;
  args::Flag report_rendering_fps;

  args::ValueFlag<int> m_max_num_frames;
  int max_num_frames() { return (m_max_num_frames) ? args::get(m_max_num_frames) : 0; }

  std::string render_mode_msg() {
    std::string msg;
    for (int i = 0; i < Items_Count(render_modes); ++i) {
      const char* out_text;
      Items_SingleStringGetter(render_modes, i, &out_text);
      msg += std::string(out_text) + "\n";
    }
    return msg;
  }

public:
  CmdArgs(const char* title, int argc, char** argv)
    : parser(title)
    , help(parser, "help", "display the help menu", {'h', "help"})
    , group1(parser, "This group is all exclusive:", args::Group::Validators::Xor)
    , group2(parser, "This group is all inclusive:", args::Group::Validators::AllOrNone)
    , m_simple_volume(group1, "filename", "the simple volume to render", {"simple-volume"})
    , m_neural_volume(group1, "filename", "the neural volume to render", {"neural-volume"})
    , m_camera_from(group2, "vec3f", "from where we are looking", {"camera-from"})
    , m_camera_at(group2, "vec3f", "which point we are looking at",   {"camera-at"})
    , m_camera_up(group2, "vec3f", "general up-vector", {"camera-up"})
    , m_tfn(parser, "filename", "the transfer function preset", {"tfn"})
    , m_sampling_rate(parser, "float", "ray marching sampling rate", {"sampling-rate"})
    , m_density_scale(parser, "float", "path tracing density scale", {"density-scale"})
    , m_rendering_mode(parser, "int", render_mode_msg(), {"rendering-mode"})
    , m_max_num_frames(parser, "int", "maximum number of frames to render", {"max-num-frames"})
    , force_camera(parser, "flag", "force the camera setting", {"force-camera"})
    , report_rendering_fps(parser, "flag", "report rendering FPS", {"report-rendering-fps"})
  {
    exec(parser, argc, argv);
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

  TransactionalValue<View> view; // produced by BG, consumed by FG
  GLuint texture = 0;

  vnrRenderer renderer;
  vnrVolume neural_volume;
  vnrTransferFunction tfn;
  vnrCamera cam;

  TransferFunctionWidget widget;

  FPSCounter fps_fg, fps_bg;
  AsyncLoop background_task;

  // renderer parameters
  TransactionalValue<TransferFunction> transfer_function;
  TransactionalValue<Camera> camera;
  TransactionalValue<float> volume_sampling_rate;
  TransactionalValue<float> volume_density_scale;
  TransactionalValue<int> rendering_mode;

  // control flows
  std::atomic<bool> frame_reset = false;
  std::atomic<bool> disable_frame_accum = false;
  std::atomic<bool> save_params = false;
  std::atomic<bool> load_params = false;
  std::atomic<bool> denoise = false;

  size_t frame_counter = 0;

  enum {
    NEURAL_VOLUME,
    SIMPLE_VOLUME
  } mode;

  CmdArgs& args;

public:
  MainWindow(CmdArgs& commandline, const std::string& title, Camera camera, const float worldScale)
    : GLFCameraWindow(title, camera.from, camera.at, camera.up, worldScale, 768, 768)
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
    glGenTextures(1, &texture);

    if (args.has_simple_volume()) {
      neural_volume = vnrCreateSimpleVolume(args.volume(), "GPU", false);
      mode = SIMPLE_VOLUME;
    }
    else {
      vnrJson params;
      vnrLoadJsonBinary(params, args.volume());
      neural_volume = vnrCreateNeuralVolume(params);
      // {
      //   for (int i = 0; i < vnrNeuralVolumeGetNumberOfBlobs(neural_volume); ++i) { 
      //     vnrNeuralVolumeDecodeProgressive(neural_volume);
      //   }
      // }
      std::cout << "[vnr] # of inference blobs = " << vnrNeuralVolumeGetNumberOfBlobs(neural_volume) << std::endl;
      mode = NEURAL_VOLUME;
    }

    // setup other components

    renderer = vnrCreateRenderer(neural_volume);

    cam = vnrCreateCamera();
    vnrCameraSet(cam, camera.from, camera.at, camera.up);
    if (args.has_simple_volume()) {
      vnrCameraSet(cam, args.volume());
    }
    if (args.has_tfn()) {
      vnrCameraSet(cam, args.tfn());
    }

    vnrRendererSetCamera(renderer, cam);

    if (args.has_tfn()) {
      tfn = vnrCreateTransferFunction(args.tfn());
    }
    else {
      tfn = vnrCreateTransferFunction();
    }
    vnrTransferFunctionSetValueRange(tfn, range1f(0, 1));
    vnrRendererSetTransferFunction(renderer, tfn);

    CUDA_SYNC_CHECK(); // sanity check

    // other params
    volume_sampling_rate = args.sampling_rate();
    volume_density_scale = args.density_scale();
    rendering_mode = args.rendering_mode();

    // setup camera
    if (!args.force_camera) {
      camera.from = vnrCameraGetPosition(cam);
      camera.at = vnrCameraGetFocus(cam);
      camera.up = vnrCameraGetUpVec(cam);
      cameraFrame.setOrientation(camera.from, camera.at, camera.up);
      // std::cout << camera.from  << std::endl;
      // std::cout << camera.at  << std::endl;
      // std::cout << camera.up  << std::endl;
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

    if (!color.empty()) {
      std::vector<vec4f> color_controls;
      for (int i = 0; i < color.size(); ++i) {
        color_controls.push_back(vec4f(i / float(color.size() - 1), /* control point position */
                                       color.at(i).x, color.at(i).y, color.at(i).z));
      }
      assert(!alpha.empty());
      widget.add_tfn(color_controls, alpha, "builtin");
    }
    widget.set_default_value_range(range.lower, range.upper);
    transfer_function.assign([&](TransferFunction& value) {
      value.color = color;
      value.alpha = alpha;
      value.range = range;
    });
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
      // std::cout << cameraFrame.get_position()  << std::endl;
      // std::cout << cameraFrame.get_poi()  << std::endl;
      // std::cout << cameraFrame.get_accurate_up()  << std::endl;
      cameraFrame.modified = false;
    }
  }

  void background_work()
  {
    TRACE_CUDA;

    if (mode == NEURAL_VOLUME) {
      if (save_params) {
        vnrNeuralVolumeSerializeParams(neural_volume, "params.json");
        save_params = false;
      }
      if (load_params) {
        vnrNeuralVolumeSetParams(neural_volume, "params.json");
        load_params = false;
      }
    }

    if (fb_size_bg.update()) {
      vnrRendererSetFramebufferSize(renderer, fb_size_bg.get());
    }

    TRACE_CUDA;
  
    if (transfer_function.update()) {
      vnrTransferFunctionSetColor(tfn, transfer_function.ref().color);
      vnrTransferFunctionSetAlpha(tfn, transfer_function.ref().alpha);
      vnrTransferFunctionSetValueRange(tfn, transfer_function.ref().range);
      vnrRendererSetTransferFunction(renderer, tfn);
    }

    TRACE_CUDA;

    if (camera.update()) {
      vnrCameraSet(cam, camera.ref().from, camera.ref().at, camera.ref().up);
      vnrRendererSetCamera(renderer, cam);
    }

    if (volume_sampling_rate.update()) {
      vnrRendererSetVolumeSamplingRate(renderer, volume_sampling_rate.get());
    }
    
    if (volume_density_scale.update()) {
      vnrRendererSetVolumeDensityScale(renderer, volume_density_scale.get());
    }

    if (rendering_mode.update()) {
      vnrRendererSetMode(renderer, rendering_mode.get());
    }

    vnrRendererSetDenoiser(renderer, denoise);

    TRACE_CUDA;

    if (frame_reset || disable_frame_accum) {
      vnrRendererResetAccumulation(renderer);
      frame_reset = false;
    }

    TRACE_CUDA;

    // rendering & training
    double time_rendering = 0., time_training = 0.;

    View view_tmp; // reference & inference view
    {
      auto t0 = std::chrono::high_resolution_clock::now();

      TRACE_CUDA;
      {
        vnrRender(renderer);
        view_tmp.pixels = vnrRendererMapFrame(renderer);
        view_tmp.size = fb_size_bg.get();
        view = view_tmp;
      }
      TRACE_CUDA;

      time_rendering = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

      ++frame_counter;
    }

    TRACE_CUDA;

    if (args.max_num_frames() > 0 && frame_counter > args.max_num_frames()) {
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
      static int gui_rendering_mode = args.rendering_mode();
      if (ImGui::Combo("Render Mode", &gui_rendering_mode, render_modes, IM_ARRAYSIZE(render_modes))) {
        rendering_mode = gui_rendering_mode;
      }

      // basic training and rendering behaviors
      if (ImGui::Button("Reset Frame")) {
        frame_reset = true;
      }

      // adjust rendering qualities
      static float gui_sampling_rate = args.sampling_rate();
      if (ImGui::SliderFloat("Volume Sampling Rate", &gui_sampling_rate, 0.01f, 10.f, "%.3f")) {
        volume_sampling_rate = gui_sampling_rate;
      }
      static float gui_density_scale = args.density_scale();
      if (ImGui::SliderFloat("Volume Density Scale", &gui_density_scale, 0.01f, 10.f, "%.3f")) {
        volume_density_scale = gui_density_scale;
      }

      static bool gui_denoise_frame = denoise;
      if (ImGui::Checkbox("Denoise", &gui_denoise_frame)) {
        denoise = gui_denoise_frame;
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

      if (mode == NEURAL_VOLUME) {
        ImGui::SameLine();
        if (ImGui::Button("Save Params")) { save_params = true; }
        ImGui::SameLine();
        if (ImGui::Button("Load Params")) { load_params = true; }
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

    glfwSetWindowTitle(handle, title.str().c_str());
  }

  void draw() override
  {
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, fb_size_fg.x, fb_size_fg.y);
    view.update([&](const View& view) { view_update(view.size, texture, view.pixels); });
    view_draw(fb_size_fg, texture);

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
    // if (!pause)
      view.access([&](const View& view) { 
        auto fname = "frame-" + filename;
        stbi_flip_vertically_on_write(1);
        saveJPG("screenshots/" + fname, view.size, view.pixels);
      });
    if (sync) background_task.start();
  }

  void resize(const vec2i& new_size) override
  {
    if (new_size.long_product() == 0) return;

    fb_size_fg.x = new_size.x;
    fb_size_fg.y = new_size.y;

    fb_size_bg = fb_size_fg;
  }

  void close()
  {
    background_task.stop();
    glDeleteTextures(1, &texture);
  }
};

extern "C" int
main(int ac, char** av)
{
  // -------------------------------------------------------
  // initialize command line arguments
  // -------------------------------------------------------
  CmdArgs args("Interactive Volume Renderer", ac, av);

  // -------------------------------------------------------
  // initialize camera
  // -------------------------------------------------------
  Camera camera = { /*from*/ args.camera_from(),
                    /* at */ args.camera_at(),
                    /* up */ args.camera_up() };

  // (C)urrent camera:
  // - from :(17.7318,-0.856024,-2158.65)
  // - poi  :(0,0,0)
  // - upVec:(-0.00111732,0.999999,-0.000405732)
  // - frame:{ vx = (-0.999966,-0.00112062,-0.00821356), vy = (-0.00111732,0.999999,-0.000405732), vz = (0.00821401,-0.000396541,-0.999966)}
  // (C)urrent camera:
  // - from :(5.35715,-0.258623,-652.174)
  // - poi  :(0,0,0)
  // - upVec:(-0.00111732,0.999999,-0.000405732)
  // - frame:{ vx = (-0.999966,-0.00112062,-0.00821356), vy = (-0.00111732,0.999999,-0.000405732), vz = (0.00821401,-0.000396541,-0.999966)}

  // something approximating the scale of the world, so the
  // camera knows how much to move for any given user interaction:
  const float worldScale = SCENE_SCALE;

  // -------------------------------------------------------
  // initialize opengl window
  // -------------------------------------------------------
  auto* window = new MainWindow(args, "Interactive Volume Renderer", camera, worldScale);

  auto t0 = std::chrono::high_resolution_clock::now();

  window->run();
  window->close();
  
  const double total = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
  std::cout << "total time = " << total << std::endl;

  delete window;

  vnrMemoryQueryPrint("memory");

  return 0;
}
