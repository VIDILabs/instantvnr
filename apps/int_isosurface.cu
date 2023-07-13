//. ======================================================================== //
//. Copyright 2019-2020 Qi Wu                                                //
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

// This file creates a renderer that only does rendering.

// clang-format off
#include <glfwapp/GLFWApp.h>
#include <glad/glad.h>
#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif // clang-format on

#include <imgui.h>
#include <implot.h>

#include <ovr/common/cross_device_buffer.h>
#include <ovr/common/vidi_async_loop.h>
#include <ovr/common/vidi_fps_counter.h>
#include <ovr/common/vidi_screenshot.h>
#include <ovr/common/vidi_transactional_value.h>
#include <ovr/renderer.h>

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

#include "cmdline.h"

#include <api.h>
#include <api_internal.h>
#include <core/marching_cube.cuh>

using namespace ovr::math;
using ovr::Camera;
using ovr::MainRenderer;

using vidi::AsyncLoop;
using vidi::FPSCounter;
using vidi::HistoryFPSCounter;
using vidi::TransactionalValue;

struct CmdArgs : CmdArgsBase {
public:
  args::ArgumentParser parser;
  args::HelpFlag help;
  args::Group group_volume;
  args::Group group_required;

  args::Positional<std::string> m_scene;
  bool has_scene() { return m_scene; }
  std::string scene() { return args::get(m_scene); }

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
    , m_scene(parser, "filename", "user-defined scene file")
    , m_simple_volume(group_volume, "filename", "the simple volume to render", {"simple-volume"})
    , m_neural_volume(group_volume, "filename", "the neural volume to render", {"neural-volume"})
    , m_isovalue(group_required, "float", "iso-value", {"iso", "isovalue"})
  {
    exec(parser, argc, argv);
  }
};

bool
extract_isosurface(vnrVolume volume, float isovalue, ovr::scene::Geometry& geometry)
{
  vec3f* verts;
  size_t n_verts;

  vnrMarchingCube(volume, isovalue, &verts, &n_verts, true);

  const size_t n_faces = n_verts / 3;

  if (n_faces == 0) return false;

  CUDABufferTyped<uint32_t> indices;
  CUDABufferTyped<vec3f> normals;
  CUDABufferTyped<vec3f> colors;

  indices.alloc(n_faces * 3);
  normals.alloc(n_faces * 3);
  colors.alloc(n_faces * 3);

  util::parallel_for_gpu(n_faces, [verts, indices=indices.d_pointer(), normals=normals.d_pointer(), colors=colors.d_pointer()] __device__ (size_t i) {
    indices[3*i+0] = 3*i+0;
    indices[3*i+1] = 3*i+1;
    indices[3*i+2] = 3*i+2;
    const vec3f A = gdt::normalize(verts[3*i+1] - verts[3*i]);
    const vec3f B = gdt::normalize(verts[3*i+2] - verts[3*i]);
    const vec3f N = gdt::cross(A, B);
    normals[3*i] = normals[3*i+1] = normals[3*i+2] = N;
    // colors[3*i] = colors[3*i+1] = colors[3*i+2] = N * 0.5f + 0.5f;
    colors[3*i] = colors[3*i+1] = colors[3*i+2] = vec3f(1,1,1);
  });

  std::vector<vec3f> h_vertices(n_verts);
  std::vector<uint32_t> h_indices;
  std::vector<vec3f> h_normals;
  std::vector<vec3f> h_colors;

  CUDA_CHECK(cudaMemcpy(h_vertices.data(), verts, n_verts * sizeof(vec3f), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(verts));

  indices.download(h_indices);
  normals.download(h_normals);
  colors.download(h_colors);
  indices.free();
  normals.free();
  colors.free();

  geometry.type = ovr::scene::Geometry::TRIANGLES_GEOMETRY;
  geometry.triangles.position = ovr::CreateArray1DFloat3(h_vertices);
  geometry.triangles.index = ovr::CreateArray1DScalar(h_indices);
  geometry.triangles.verts.normal = ovr::CreateArray1DFloat3(h_normals);
  geometry.triangles.verts.color = ovr::CreateArray1DFloat3(h_colors);

  return true;
}

struct MainWindow : public glfwapp::GLFCameraWindow {  
private:
  std::shared_ptr<MainRenderer> renderer;
  MainRenderer::FrameBufferData renderer_output;

  TransactionalValue<float> isovalue{ 0.5f };
  TransactionalValue<bool> path_tracing{ false };

  struct FrameOutputs {
    vec2i size{ 0 };
    vec4f* rgba{ nullptr };
  };

  TransactionalValue<FrameOutputs> frame_outputs; /* wrote by BG, consumed by GUI */
  GLuint frame_texture{ 0 };                        /* local to GUI thread */
  vec2i frame_size_local{ 0 };                      /* local to GUI thread */
  TransactionalValue<vec2i> frame_size_shared{ 0 }; /* wrote by GUI, consumed by BG */
  double frame_time = 0.0;

  vnrVolume volume;

  /* local to GUI thread */
  bool async_enabled{ true }; /* local to GUI thread */
  AsyncLoop async_rendering_loop;

  std::atomic<float> variance{ 0 }; /* not critical */
  HistoryFPSCounter foreground_fps; /* thread safe */
  HistoryFPSCounter background_fps;

  bool gui_enabled{ true }; /* trigger GUI to show */
  bool gui_performance_enabled{ true }; /* trigger performance GUI to show */

public:
  MainWindow(const std::string& title,
             std::shared_ptr<MainRenderer> renderer,
             vnrVolume volume,
             const Camera& camera,
             const float scale,
             int width,
             int height)
    : GLFCameraWindow(title, camera.from, camera.at, camera.up, scale, width, height)
    , async_rendering_loop(std::bind(&MainWindow::render_background, this))
    , renderer(renderer)
    , volume(volume)
  {
    ImPlot::CreateContext();

    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glGenTextures(1, &frame_texture);

    resize(vec2i(0, 0));

    if (async_enabled) {
      render_background(); // warm up
      async_rendering_loop.start();
    }
  }

  ~MainWindow()
  {
    ImPlot::DestroyContext();
  }

  /* background thread */
  void render_background()
  {
    auto start = std::chrono::high_resolution_clock::now(); 

    if (frame_size_shared.update()) {
      frame_outputs.assign([&](FrameOutputs& d) { d.size = vec2i(0, 0); });
      renderer->set_fbsize(frame_size_shared.ref());
    }
    if (frame_size_shared.ref().long_product() == 0) return;

    if (path_tracing.update()) {
      renderer->set_path_tracing(path_tracing.get());
    }
  
    if (isovalue.update()) {
      auto& models = renderer->current_scene.instances[0].models;
      ovr::scene::Geometry geometry;
      if (extract_isosurface(volume, isovalue.get(), geometry)) {
        models.resize(1);
        models[0].type = ovr::scene::Model::GEOMETRIC_MODEL;
        models[0].geometry_model.geometry = geometry;
        renderer->init(0, NULL, renderer->current_scene, Camera{ cameraFrame.get_position(), cameraFrame.get_poi(), cameraFrame.get_accurate_up() });
      }
      else {
        models.clear();
        renderer->init(0, NULL, renderer->current_scene, Camera{ cameraFrame.get_position(), cameraFrame.get_poi(), cameraFrame.get_accurate_up() });
      }
    }

    renderer->commit();
    renderer->mapframe(&renderer_output);

    FrameOutputs output;
    {
      output.rgba = (vec4f*)renderer_output.rgba->to_cpu()->data();
      output.size = frame_size_shared.get();
    }
    frame_outputs = output;

    renderer->swap();

    variance = renderer->unsafe_get_variance();

    double render_time = 0.0;
    renderer->render(); 
    render_time = renderer->render_time; 

    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    frame_time += diff.count();

    if (background_fps.count()) {
      background_fps.update_history((float)frame_time / 10.f, (float)render_time / 10.f, 0.0f);
      renderer->render_time = 0.0;
      frame_time = 0.0;
    }
  }

  /* GUI thread */
  void render() override
  {
    if (cameraFrame.modified) {
      renderer->set_camera(cameraFrame.get_position(), cameraFrame.get_poi(), cameraFrame.get_accurate_up());
      cameraFrame.modified = false;
    }

    if (!async_enabled) {
      render_background();
    }
  }

  /* GUI thread */
  virtual void key(int key, int mods) override
  {
    switch (key)
    {
      case 'f':
      case 'F':
        std::cout << "Entering 'fly' mode" << std::endl;
        if (flyModeManip)
          cameraFrameManip = flyModeManip;
        break;
      case 'i':
      case 'I':
        std::cout << "Entering 'inspect' mode" << std::endl;
        if (inspectModeManip)
          cameraFrameManip = inspectModeManip;
        break;
      case 'g':
      case 'G':
        std::cout << "Toggling GUI" << std::endl;
        gui_enabled = !gui_enabled;
        break;
      case 'p':
      case 'P':
        std::cout << "Toggling performance GUI" << std::endl;
        gui_performance_enabled = !gui_performance_enabled;
        break;
      case 's':
      case 'S':
        std::cout << "Saving screenshot" << std::endl;
        {
          const FrameOutputs& out = frame_outputs.get();
          vidi::Screenshot::save(out.rgba, out.size);
        }
        break;
      case GLFW_KEY_ESCAPE:
        glfwSetWindowShouldClose(GLFWindow::handle, GLFW_TRUE);
      default:
        if (cameraFrameManip)
          cameraFrameManip->key(key, mods);
    }
  }

  /* GUI thread */
  void set_transfer_function(const std::vector<vec3f>& c, const std::vector<vec2f>& o, const vec2f& r)
  {
    std::vector<float> cc(c.size() * 3);
    for (int i = 0; i < c.size(); ++i) {
      cc[3 * i + 0] = c[i].x;
      cc[3 * i + 1] = c[i].y;
      cc[3 * i + 2] = c[i].z;
    }
    std::vector<float> oo(o.size() * 2);
    for (int i = 0; i < o.size(); ++i) {
      oo[2 * i + 0] = o[i].x;
      oo[2 * i + 1] = o[i].y;
    }
    renderer->set_transfer_function(cc, oo, r);
  }

  /* GUI thread */
  void draw() override
  {
    glBindTexture(GL_TEXTURE_2D, frame_texture);

    frame_outputs.update([&](const FrameOutputs& out) {
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, out.size.x, out.size.y, 0, GL_RGBA, GL_FLOAT, out.rgba);
    });

    const auto& size = frame_size_local;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(1, 1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, frame_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, size.x, size.y);
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

    if (gui_enabled) {
      if (ImGui::Begin("Control Panel", NULL)) {
        static float iso = 0.0f;
        if (ImGui::SliderFloat("Iso Value", &iso, 0.0f, 1.0f, "%.3f")) {
          isovalue = iso;
        }
        static bool pt = true;
        if (ImGui::Checkbox("Path Tracing", &pt)) {
          path_tracing = pt;
        }
      }
      ImGui::End();
    }

    // Performance Graph
    if (gui_performance_enabled) {
      ImGui::SetNextWindowPos(ImVec2(/*padding*/2.0f, 2.0f));
      ImGui::SetNextWindowSizeConstraints(ImVec2(300, 200), ImVec2(FLT_MAX, FLT_MAX));
      if (ImGui::Begin("Performance", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration)) {
        if (ImPlot::BeginPlot("##Performance Plot", ImVec2(500,150), ImPlotFlags_AntiAliased | ImPlotFlags_NoFrame)) {
          ImPlot::SetupAxes("frame history", "time [ms]", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
          ImPlot::PlotLine("frame time", background_fps.indices.data(), background_fps.frame_time_history.data(), (int)background_fps.frame_time_history.size());
          ImPlot::EndPlot();
        }
      }
      ImGui::End();
    }

    if (foreground_fps.count()) {
      std::stringstream title;
      title << std::fixed << std::setprecision(3) << " fg = " << foreground_fps.fps << " fps,";
      title << std::fixed << std::setprecision(3) << " bg = " << background_fps.fps << " fps,";
      title << std::fixed << std::setprecision(3) << " var = " << variance << ".";
      glfwSetWindowTitle(handle, title.str().c_str());
    }
  }

  /* GUI thread */
  void resize(const vec2i& size) override
  {
    frame_size_local = size;
    frame_size_shared = size;
  }

  /* GUI thread */
  void close() override
  {
    if (async_enabled)
      async_rendering_loop.stop();

    glDeleteTextures(1, &frame_texture);
  }
};

/*! main entry point to this example - initially optix, print hello world, then exit */
int
main(int ac, const char** av)
{
  ovr::Scene scene;

  // -------------------------------------------------------
  // initialize command line arguments
  // -------------------------------------------------------
  CmdArgs args("Commandline Volume Renderer", ac, (char**)av);

  vnrVolume volume;
  if (args.has_simple_volume()) {
    volume = vnrCreateSimpleVolume(args.volume(), "GPU", false);
  }
  else {
    vnrJson params;
    vnrLoadJsonBinary(params, args.volume());
    volume = vnrCreateNeuralVolume(params);
  }

  vec3f scene_center = vec3f(volume->desc.dims) * 0.5f;
  box3f scene_bounds = box3f(vec3f(0), vec3f(volume->desc.dims));

  ovr::scene::Model model;
  model.type = ovr::scene::Model::GEOMETRIC_MODEL;
  extract_isosurface(volume, args.isovalue(), model.geometry_model.geometry);
 
  ovr::scene::Instance instance;
  instance.models.push_back(model);
  instance.transform = affine3f::translate(-scene_center);

  scene.instances.push_back(instance);

  // if (args.has_scene()) {
  //   scene = ovr::scene::create_scene(args.scene());
  // }
  // else {
  //   scene = create_example_scene();
  //   scene.camera = { /*from*/ vec3f(0.f, 0.f, -1200.f),
  //                    /* at */ vec3f(0.f, 0.f, 0.f),
  //                    /* up */ vec3f(0.f, 1.f, 0.f) };
  // }

  scene.camera = { 
    /*from*/ vec3f(0.f, 0.f, -1200.f),
    /* at */ vec3f(0.f, 0.f, 0.f),
    /* up */ vec3f(0.f, 1.f, 0.f) 
  };

  std::shared_ptr<ovr::MainRenderer> renderer = create_renderer("ospray");
  renderer->set_path_tracing(true);
  renderer->set_frame_accumulation(true);
  renderer->set_sparse_sampling(false);
  renderer->init(ac, av, scene, scene.camera);

  // -------------------------------------------------------
  // initialize opengl window
  // -------------------------------------------------------
  MainWindow* window = new MainWindow("VNR", renderer, volume, scene.camera, 100.f, 800, 800);
  window->run();
  window->close();

  return 0;
}
