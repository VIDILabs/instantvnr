#pragma once

#include "../instantvnr_types.h"

#include <string>
#include <vector>
#include <map>

namespace vnr {

// ------------------------------------------------------------------
// Object Definitions
// ------------------------------------------------------------------

enum ObjectType {
  VOLUME_STRUCTURED_REGULAR,
};

#ifndef __NVCC__
static std::string
object_type_string(ObjectType t)
{
  switch (t) {
  case VOLUME_STRUCTURED_REGULAR: return "VOLUME_STRUCTURED_REGULAR";
  default: throw std::runtime_error("unknown object type");
  }
}
#endif

enum {
  VISIBILITY_MASK_GEOMETRY = 0x1,
  VISIBILITY_MASK_VOLUME = 0x2,
};


struct OptixProgram
{
public:
  struct HitGroupShaders
  {
    std::string shader_CH;
    std::string shader_AH;
    std::string shader_IS;
  };

  struct ObjectGroup
  {
    ObjectType type;
    std::vector<HitGroupShaders> hitgroup;
  };

  struct InstanceHandler
  {
    ObjectType type;  // object type
    uint32_t idx = 0; // object index
    OptixInstance handler;
  };

  OptixProgram(std::string ptx_code, uint32_t num_ray_types) : ptx_code(ptx_code), num_ray_types(num_ray_types) {}

  virtual ~OptixProgram()
  {
    program.raygen_buffer.free(0);
    program.miss_buffer.free(0);
    program.hitgroup_buffer.free(0);
  }

  void init(OptixDeviceContext, std::map<ObjectType, std::vector<void*>> records, std::vector<std::vector<OptixProgram::InstanceHandler>> blas);

protected:
  /*! creates the module that contains all the programs we are going
    to use. in this simple example, we use a single module from a
    single .cu file, using a single embedded ptx string */
  void createModule(OptixDeviceContext);

  /*! does all setup for the raygen program(s) we are going to use */
  void createRaygenPrograms(OptixDeviceContext);

  /*! does all setup for the miss program(s) we are going to use */
  void createMissPrograms(OptixDeviceContext);

  /*! does all setup for the hitgroup program(s) we are going to use */
  void createHitgroupPrograms(OptixDeviceContext);

  /*! assembles the full pipeline of all programs */
  void createPipeline(OptixDeviceContext);

  /*! constructs the shader binding table */
  void createSBT(OptixDeviceContext, const std::map<ObjectType, std::vector<void*>>&);

  /*! build the top level acceleration structure */
  void createTLAS(OptixDeviceContext, const std::vector<std::vector<OptixProgram::InstanceHandler>>&);

protected:
  /*! @{ the pipeline we're building */
  struct
  {
    OptixPipeline handle{};
    OptixPipelineCompileOptions compile_opts{};
    OptixPipelineLinkOptions link_opts{};
  } pipeline;
  /*! @} */

  /*! @{ the module that contains out device programs */
  struct
  {
    OptixModule handle{};
    OptixModuleCompileOptions compile_opts{};
  } module;
  /*! @} */

  /*! @{ vector of all our program(group)s, and the SBT built around them */
  struct
  {
    std::vector<OptixProgramGroup> raygens;
    CUDABuffer raygen_buffer;
    std::vector<OptixProgramGroup> misses;
    CUDABuffer miss_buffer;
    std::vector<OptixProgramGroup> hitgroups;
    CUDABuffer hitgroup_buffer;
  } program;

  OptixShaderBindingTable sbt = {};
  /*! @} */

  std::string shader_raygen;
  std::vector<std::string> shader_misses;
  std::vector<ObjectGroup> shader_objects;

  std::map<ObjectType, std::vector<uint32_t>> sbt_offset_table;

  struct IasData
  {
    std::vector<OptixInstance> instances; /*! the ISA handlers */
    CUDABuffer instances_buffer;          /*! one buffer for all ISAs on GPU */
    CUDABuffer as_buffer;                 /*! buffer that keeps the (final, compacted) accel structure */
    OptixTraversableHandle traversable;   // <- the output

    ~IasData()
    {
      instances_buffer.free(0);
      as_buffer.free(0);
    }
  };
  std::shared_ptr<IasData[]> ias;

  const std::string ptx_code;
  const uint32_t num_ray_types;
};

struct OptixProgramDenoiser
{
private:
  OptixDenoiser denoiser = nullptr;
  CUDABuffer denoiserScratch;
  CUDABuffer denoiserState;

  CUDABuffer inputBuffer;

public:
  vec4f* d_ptr() const 
  {
    return (vec4f*)inputBuffer.d_pointer();
  }

  void process(OptixDeviceContext optixContext, cudaStream_t stream, bool accumulate, const uint32_t& frameID, const vec2i& frameSize, vec4f* output)
  {
    OptixDenoiserParams denoiserParams;
    denoiserParams.denoiseAlpha = 1;
    denoiserParams.hdrIntensity = (CUdeviceptr)0;
    if (accumulate)
        denoiserParams.blendFactor = 1.f / (frameID);
    else
        denoiserParams.blendFactor = 0.0f;
  
    // -------------------------------------------------------
    OptixImage2D inputLayer;
    inputLayer.data = inputBuffer.d_pointer();
    /// Width of the image (in pixels)
    inputLayer.width = frameSize.x;
    /// Height of the image (in pixels)
    inputLayer.height = frameSize.y;
    /// Stride between subsequent rows of the image (in bytes).
    inputLayer.rowStrideInBytes = frameSize.x * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    inputLayer.pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    inputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // -------------------------------------------------------
    OptixImage2D outputLayer;
    outputLayer.data = (CUdeviceptr)output;
    /// Width of the image (in pixels)
    outputLayer.width = frameSize.x;
    /// Height of the image (in pixels)
    outputLayer.height = frameSize.y;
    /// Stride between subsequent rows of the image (in bytes).
    outputLayer.rowStrideInBytes = frameSize.x * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    outputLayer.pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

#if OPTIX_VERSION >= 70300
    OptixDenoiserGuideLayer denoiserGuideLayer = {};

    OptixDenoiserLayer denoiserLayer = {};
    denoiserLayer.input = inputLayer;
    denoiserLayer.output = outputLayer;

    OPTIX_CHECK(optixDenoiserInvoke(denoiser,
                                    /*stream*/ stream,
                                    &denoiserParams,
                                    denoiserState.d_pointer(),
                                    denoiserState.sizeInBytes,
                                    &denoiserGuideLayer,
                                    &denoiserLayer,
                                    1,
                                    /*inputOffsetX*/ 0,
                                    /*inputOffsetY*/ 0,
                                    denoiserScratch.d_pointer(),
                                    denoiserScratch.sizeInBytes));
#else
    OPTIX_CHECK(optixDenoiserInvoke(denoiser,
                                    /*stream*/ stream,
                                    &denoiserParams,
                                    denoiserState.d_pointer(),
                                    denoiserState.sizeInBytes,
                                    &inputLayer,
                                    1,
                                    /*inputOffsetX*/ 0,
                                    /*inputOffsetY*/ 0,
                                    &outputLayer,
                                    denoiserScratch.d_pointer(),
                                    denoiserScratch.sizeInBytes));
#endif
  }

  void resize(OptixDeviceContext optixContext, cudaStream_t stream, const vec2i& newSize)
  {
    if (denoiser) {
      OPTIX_CHECK(optixDenoiserDestroy(denoiser));
    };

    inputBuffer.resize(newSize.x * newSize.y * sizeof(vec4f), stream);

    // ------------------------------------------------------------------
    // create the denoiser:
    OptixDenoiserOptions denoiserOptions = {};

#if OPTIX_VERSION >= 70300
    OPTIX_CHECK(optixDenoiserCreate(optixContext, OPTIX_DENOISER_MODEL_KIND_LDR, &denoiserOptions, &denoiser));
#else
    denoiserOptions.inputKind = OPTIX_DENOISER_INPUT_RGB;

#if OPTIX_VERSION < 70100
    // these only exist in 7.0, not 7.1
    denoiserOptions.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
#endif
    OPTIX_CHECK(optixDenoiserCreate(optixContext, &denoiserOptions, &denoiser));
    OPTIX_CHECK(optixDenoiserSetModel(denoiser, OPTIX_DENOISER_MODEL_KIND_LDR, NULL, 0));
#endif

    // .. then compute and allocate memory resources for the denoiser
    OptixDenoiserSizes denoiserReturnSizes;
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(denoiser, newSize.x, newSize.y, &denoiserReturnSizes));

#if OPTIX_VERSION < 70100
    denoiserScratch.resize(denoiserReturnSizes.recommendedScratchSizeInBytes, stream);
#else
    denoiserScratch.resize(std::max(denoiserReturnSizes.withOverlapScratchSizeInBytes, denoiserReturnSizes.withoutOverlapScratchSizeInBytes), stream);
#endif
    denoiserState.resize(denoiserReturnSizes.stateSizeInBytes, stream);

    // ------------------------------------------------------------------
    OPTIX_CHECK(optixDenoiserSetup(denoiser, stream, newSize.x, newSize.y,
                                   denoiserState.d_pointer(),
                                   denoiserState.sizeInBytes,
                                   denoiserScratch.d_pointer(),
                                   denoiserScratch.sizeInBytes));
  }
};

} // namespace vnr
