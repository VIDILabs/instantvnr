//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //

#pragma once

#include "core/mathdef.h"

#include <json/json.hpp>

#include <vector>
#include <memory>

namespace vnr {
using json = nlohmann::json;
struct Camera;
struct TransferFunction;
struct VolumeContext;
struct RenderContext;
}

typedef std::shared_ptr<vnr::Camera> vnrCamera;
typedef std::shared_ptr<vnr::TransferFunction> vnrTransferFunction;

typedef vnr::ValueType vnrType;
typedef std::shared_ptr<vnr::VolumeContext> vnrVolume;
typedef std::shared_ptr<vnr::RenderContext> vnrRenderer;

enum vnrRenderMode {
  // reference ray marcher implmented in optix
  VNR_OPTIX_NO_SHADING = 0,
  VNR_OPTIX_GRADIENT_SHADING,
  VNR_OPTIX_FULL_SHADOW,
  VNR_OPTIX_SINGLE_SHADE_HEURISTIC,
  // ray marching with local phong shading
  VNR_RAYMARCHING_NO_SHADING_DECODING,
  VNR_RAYMARCHING_NO_SHADING_SAMPLE_STREAMING,
  VNR_RAYMARCHING_NO_SHADING_IN_SHADER,
  // ray marching with local phong shading
  VNR_RAYMARCHING_GRADIENT_SHADING_DECODING,
  VNR_RAYMARCHING_GRADIENT_SHADING_SAMPLE_STREAMING,
  VNR_RAYMARCHING_GRADIENT_SHADING_IN_SHADER,
  // ray marching with single-shot gradient shading
  VNR_RAYMARCHING_SINGLE_SHADE_HEURISTIC_DECODING,
  VNR_RAYMARCHING_SINGLE_SHADE_HEURISTIC_SAMPLE_STREAMING,
  VNR_RAYMARCHING_SINGLE_SHADE_HEURISTIC_IN_SHADER,
  // path tracing based global illumination
  VNR_PATHTRACING_DECODING,
  VNR_PATHTRACING_SAMPLE_STREAMING,
  VNR_PATHTRACING_IN_SHADER,
  // terminal
  VNR_INVALID,
};

inline bool vnrRequireDecoding(int m) 
{
  switch ((vnrRenderMode)m) {
  case VNR_OPTIX_NO_SHADING:
  case VNR_OPTIX_GRADIENT_SHADING:
  case VNR_OPTIX_FULL_SHADOW:
  case VNR_OPTIX_SINGLE_SHADE_HEURISTIC: return true;

  case VNR_RAYMARCHING_NO_SHADING_DECODING: return true;
  case VNR_RAYMARCHING_NO_SHADING_SAMPLE_STREAMING: 
  case VNR_RAYMARCHING_NO_SHADING_IN_SHADER: return false;
  
  case VNR_RAYMARCHING_GRADIENT_SHADING_DECODING: return true;
  case VNR_RAYMARCHING_GRADIENT_SHADING_SAMPLE_STREAMING: 
  case VNR_RAYMARCHING_GRADIENT_SHADING_IN_SHADER: return false;
  
  case VNR_RAYMARCHING_SINGLE_SHADE_HEURISTIC_DECODING: return true;
  case VNR_RAYMARCHING_SINGLE_SHADE_HEURISTIC_SAMPLE_STREAMING: 
  case VNR_RAYMARCHING_SINGLE_SHADE_HEURISTIC_IN_SHADER: return false;
  
  case VNR_PATHTRACING_DECODING: return true;
  case VNR_PATHTRACING_SAMPLE_STREAMING:
  case VNR_PATHTRACING_IN_SHADER: return false;

  default: throw std::runtime_error("unknown rendering mode");
  }
}

using vnrJson = vnr::json;
vnrJson vnrCreateJsonText  (std::string filename);
vnrJson vnrCreateJsonBinary(std::string filename);
void vnrLoadJsonText  (vnrJson&, std::string filename);
void vnrLoadJsonBinary(vnrJson&, std::string filename);
void vnrSaveJsonText  (const vnrJson&, std::string filename);
void vnrSaveJsonBinary(const vnrJson&, std::string filename);


// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

vnrCamera vnrCreateCamera();
vnrCamera vnrCreateCamera(const vnrJson& scene);
void vnrCameraSet(vnrCamera, vnr::vec3f from, vnr::vec3f at, vnr::vec3f up);
void vnrCameraSet(vnrCamera self, const vnrJson& scene);

vnr::vec3f vnrCameraGetPosition(vnrCamera);
vnr::vec3f vnrCameraGetFocus(vnrCamera);
vnr::vec3f vnrCameraGetUpVec(vnrCamera);

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

// simple volume
vnrVolume vnrCreateSimpleVolume(const void* data, vnr::vec3i dims, std::string type, vnr::range1f range, std::string mode);
vnrVolume vnrCreateSimpleVolume(const vnrJson& scene, std::string mode, bool save_loaded_volume = false);
void vnrSimpleVolumeSetCurrentTimeStep(vnrVolume, int time);
int  vnrSimpleVolumeGetNumberOfTimeSteps(vnrVolume);

// neural volume
vnrVolume vnrCreateNeuralVolume(const vnrJson& config, vnrVolume groundtruth, bool online_macrocell_construction = true, size_t batchsize = 1 << 16);
vnrVolume vnrCreateNeuralVolume(const vnrJson& config, vnr::vec3i dims, size_t batchsize = 1 << 16);
vnrVolume vnrCreateNeuralVolume(const vnrJson& params, size_t batchsize = 1 << 16);

void vnrNeuralVolumeSetModel (vnrVolume, const vnrJson& config);
void vnrNeuralVolumeSetParams(vnrVolume, const vnrJson& params);

double vnrNeuralVolumeGetMSE(vnrVolume, bool verbose);
double vnrNeuralVolumeGetPSNR(vnrVolume, bool verbose);
double vnrNeuralVolumeGetSSIM(vnrVolume, bool verbose);
double vnrNeuralVolumeGetTestingLoss(vnrVolume);
double vnrNeuralVolumeGetTrainingLoss(vnrVolume);
int    vnrNeuralVolumeGetTrainingStep(vnrVolume);
int    vnrNeuralVolumeGetNumberOfBlobs(vnrVolume);

int vnrNeuralVolumeGetNBytesMultilayerPerceptron(vnrVolume);
int vnrNeuralVolumeGetNBytesEncoding(vnrVolume);

void vnrNeuralVolumeTrain(vnrVolume, int steps, bool fast_mode, bool verbose = false);
void vnrNeuralVolumeDecodeProgressive(vnrVolume);

void vnrNeuralVolumeDecode(vnrVolume, float* output);
void vnrNeuralVolumeDecodeInference(vnrVolume, std::string filename);
void vnrNeuralVolumeDecodeReference(vnrVolume, std::string filename);

void vnrNeuralVolumeSerializeParams(vnrVolume, std::string filename);
void vnrNeuralVolumeSerializeParams(vnrVolume, vnrJson& params);

// general
void vnrVolumeSetClippingBox(vnrVolume, vnr::vec3f lower, vnr::vec3f upper);
void vnrVolumeSetScaling(vnrVolume, vnr::vec3f scale);
vnr::range1f vnrVolumeGetValueRange(vnrVolume);

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------
struct vnrIsosurface {
  float isovalue;   // input
  vnr::vec3f** ptr; // output
  size_t* size;     // output
  double et = 0.0;   // output
};
double vnrMarchingCube(vnrVolume volume, float isovalue, vnr::vec3f** ptr, size_t* size, bool cuda);
void vnrMarchingCube(vnrVolume volume, vnrIsosurface& isosurface, bool output_to_cuda_memory);
void vnrMarchingCube(vnrVolume volume, std::vector<vnrIsosurface>& isosurfaces, bool output_to_cuda_memory);
void vnrSaveTriangles(std::string filename, const vnr::vec3f* ptr, size_t size);

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

vnrTransferFunction vnrCreateTransferFunction();
vnrTransferFunction vnrCreateTransferFunction(const vnrJson& scene);
void vnrTransferFunctionSetColor(vnrTransferFunction, const std::vector<vnr::vec3f>& colors);
void vnrTransferFunctionSetAlpha(vnrTransferFunction, const std::vector<vnr::vec2f>& alphas);
void vnrTransferFunctionSetValueRange(vnrTransferFunction, vnr::range1f range);

const std::vector<vnr::vec3f>& vnrTransferFunctionGetColor(vnrTransferFunction);
const std::vector<vnr::vec2f>& vnrTransferFunctionGetAlpha(vnrTransferFunction);
const vnr::range1f& vnrTransferFunctionGetValueRange(vnrTransferFunction);

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

vnrRenderer vnrCreateRenderer(vnrVolume);
void vnrRendererSetFramebufferSize(vnrRenderer, vnr::vec2i fbsize);
void vnrRendererSetTransferFunction(vnrRenderer, vnrTransferFunction);
void vnrRendererSetCamera(vnrRenderer, vnrCamera);
void vnrRendererSetMode(vnrRenderer, int mode);
void vnrRendererSetDenoiser(vnrRenderer self, bool enable_or_not);
void vnrRendererSetVolumeSamplingRate(vnrRenderer self, float value);
void vnrRendererSetVolumeDensityScale(vnrRenderer self, float value);
void vnrRendererResetAccumulation(vnrRenderer);
void vnrRender(vnrRenderer);
vnr::vec4f* vnrRendererMapFrame(vnrRenderer);


// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

void vnrResetMaxMemory();
void vnrMemoryQuery(size_t* used_by_self, size_t* used_by_tcnn, size_t* used_peak, size_t* used_total);
void vnrMemoryQueryPrint(const char* prompt);
void vnrFreeTemporaryGPUMemory();
void vnrCompilationStatus(const char* prompt);
