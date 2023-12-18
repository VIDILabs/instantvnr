#include "instantvnr_types.h"

INSTANT_VNR_NAMESPACE_BEGIN

void
TransferFunctionObject::clean()
{
  if (tfn_color_array_handler) {
    CUDA_CHECK_NOEXCEPT(cudaTrackedFreeArray(tfn_color_array_handler));
    tfn_color_array_handler = NULL;
  }
  if (tfn.colors.data) {
    CUDA_CHECK_NOEXCEPT(cudaDestroyTextureObject(tfn.colors.data));
    tfn.colors.data = { 0 };
  }
  if (tfn.colors.rawptr) {
    CUDA_CHECK_NOEXCEPT(cudaTrackedFree(tfn.colors.rawptr, tfn.colors.length * sizeof(float4)));
    tfn.colors.rawptr = nullptr;
  }
  tfn.colors.length = 0;

  if (tfn_alpha_array_handler) {
    CUDA_CHECK_NOEXCEPT(cudaTrackedFreeArray(tfn_alpha_array_handler));
    tfn_color_array_handler = NULL;
  }
  if (tfn.alphas.data) {
    CUDA_CHECK_NOEXCEPT(cudaDestroyTextureObject(tfn.alphas.data));
    tfn.alphas.data = { 0 };
  }
  if (tfn.alphas.rawptr) {
    CUDA_CHECK_NOEXCEPT(cudaTrackedFree(tfn.alphas.rawptr, tfn.alphas.length * sizeof(float)));
    tfn.alphas.rawptr = nullptr;
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

  if (!colors_data.empty()) {
    CreateArray1DFloat4(stream, colors_data, tfn_color_array_handler, tfn.colors);
  }

  TRACE_CUDA;

  if (!alphas_data.empty()) {
    CreateArray1DScalar(stream, alphas_data, tfn_alpha_array_handler, tfn.alphas);
  }

  TRACE_CUDA;
}

void 
TransferFunctionObject::update(const TransferFunction& input, const range1f original_data_range, cudaStream_t stream)
{
  const std::vector<vec3f>& c = input.color;
  const std::vector<vec2f>& o = input.alpha;
  range1f r = input.range;
  if (!r.is_empty()) {
    r.upper = min(original_data_range.upper, r.upper);
    r.lower = max(original_data_range.lower, r.lower);
  }
  set_transfer_function(c, o, r, stream);
}

INSTANT_VNR_NAMESPACE_END
