#include "instantvnr_types.h"

INSTANT_VNR_NAMESPACE_BEGIN

void
TransferFunctionObject::clean()
{
  if (tfn_color_array_handler) {
    CUDA_CHECK_NOEXCEPT(cudaFreeArray(tfn_color_array_handler));
    tfn_color_array_handler = NULL;
    util::total_n_bytes_allocated() -= tfn.colors.length * sizeof(float4);
#ifdef VNR_VERBOSE_MEMORY_ALLOCS
    printf("[mem] Array1D free %s\n", util::prettyBytes(tfn.colors.length * sizeof(float4)).c_str());
#endif
  }
  if (tfn.colors.data) {
    CUDA_CHECK_NOEXCEPT(cudaDestroyTextureObject(tfn.colors.data));
    tfn.colors.data = { 0 };
  }
  if (tfn.colors.rawptr) {
    CUDA_CHECK_NOEXCEPT(cudaFree(tfn.colors.rawptr));
    tfn.colors.rawptr = nullptr;
    util::total_n_bytes_allocated() -= tfn.colors.length * sizeof(float4);
#ifdef VNR_VERBOSE_MEMORY_ALLOCS
    printf("[mem] Linear free %s\n", util::prettyBytes(tfn.colors.length * sizeof(float4)).c_str());
#endif
  }
  tfn.colors.length = 0;

  if (tfn_alpha_array_handler) {
    CUDA_CHECK_NOEXCEPT(cudaFreeArray(tfn_alpha_array_handler));
    tfn_color_array_handler = NULL;
    util::total_n_bytes_allocated() -= tfn.alphas.length * sizeof(float);
#ifdef VNR_VERBOSE_MEMORY_ALLOCS
    printf("[mem] Array1D free %s\n", util::prettyBytes(tfn.alphas.length * sizeof(float)).c_str());
#endif
  }
  if (tfn.alphas.data) {
    CUDA_CHECK_NOEXCEPT(cudaDestroyTextureObject(tfn.alphas.data));
    tfn.alphas.data = { 0 };
  }
  if (tfn.alphas.rawptr) {
    CUDA_CHECK_NOEXCEPT(cudaFree(tfn.alphas.rawptr));
    tfn.alphas.rawptr = nullptr;
    util::total_n_bytes_allocated() -= tfn.alphas.length * sizeof(float);
#ifdef VNR_VERBOSE_MEMORY_ALLOCS
    printf("[mem] Linear free %s\n", util::prettyBytes(tfn.alphas.length * sizeof(float)).c_str());
#endif
  }
  tfn.alphas.length = 0;
}

void 
TransferFunctionObject::set_transfer_function(const std::vector<vec3f>& c, const std::vector<vec2f>& o, const range1f& r, cudaStream_t stream)
{
  std::vector<float4> colors_data;
  std::vector<float> alphas_data;
  colors_data.resize(c.size());
  for (int i = 0; i < colors_data.size(); ++i) {
    colors_data[i].x = c[i].x;
    colors_data[i].y = c[i].y;
    colors_data[i].z = c[i].z;
    colors_data[i].w = 1.f;
  }
  alphas_data.resize(o.size());
  for (int i = 0; i < alphas_data.size(); ++i) {
    alphas_data[i] = o[i].y;
  }

  TRACE_CUDA;

  tfn.range = r;
  tfn.range_rcp_norm = 1.f / tfn.range.span();

  TRACE_CUDA;

  if (!colors_data.empty())
    CreateArray1DFloat4(stream, colors_data, tfn_color_array_handler, tfn.colors);
  
  TRACE_CUDA;

  if (!alphas_data.empty())
    CreateArray1DScalar(stream, alphas_data, tfn_alpha_array_handler, tfn.alphas);

  TRACE_CUDA;
}

INSTANT_VNR_NAMESPACE_END
