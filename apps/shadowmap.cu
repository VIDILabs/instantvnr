// TODO this file is incomplete

#include "cmdline.h"

#include "device/device_nnvolume_array.h"

#include <ovr/common/math_def.h>
#include <ovr/common/random/random.h>
#include <ovr/common/dylink/Library.h>
#include <ovr/scene.h>
#include <ovr/serializer/serializer.h>

#include <cuda/cuda_buffer.h>

#define TFN_MODULE_EXTERNAL_VECTOR_TYPES
namespace tfn {
typedef ovr::math::vec2f vec2f;
typedef ovr::math::vec2i vec2i;
typedef ovr::math::vec3f vec3f;
typedef ovr::math::vec3i vec3i;
typedef ovr::math::vec4f vec4f;
typedef ovr::math::vec4i vec4i;
} // namespace tfn
#include <tfn/core.h>

#include <iostream>
#include <fstream>
#include <random>

#define inf float_large
#define float_large 1e31f
#define float_small 1e-31f
#define nearly_one 0.9999f


// to avoid crazy formatting and indentations
#define NAMESPACE_BEGIN    \
    namespace ovr          \
    {                      \
        namespace nnvolume \
        {
#define NAMESPACE_END \
    }                 \
    }


NAMESPACE_BEGIN

using namespace ovr::math;
using ovr::random::RandomTEA;

struct Light
{
    vec3f direction;
    float intensity;
};

struct DeviceVolume
{
    Array3DScalarCUDA volume;
    DeviceTransferFunction tfn;
    float step = 1.f;
    float step_rcp = 1.f; // GPU cacne to avoid recomputation

    box3f bbox = box3f(vec3f(0), vec3f(1)); // with respect to [0-1]^3
    affine3f transform;

    int n_lights = 0;
    Light *lights{ nullptr };

    int ao_samples = 1;
};

struct HostVolume
{
    DeviceVolume device;
    CUDABuffer device_buffer;

    affine3f matrix;
    
    std::vector<vec4f> tfn_colors_data;
    std::vector<float> tfn_alphas_data;
    vec2f original_value_range;

    std::vector<Light> lights;
    CUDABuffer lights_buffer;

public:
    void load_from_array3d_scalar(array_3d_scalar_t array, float data_value_min = 1, float data_value_max = -1)
    {
        Array3DScalarCUDA &output = device.volume;
        output = CreateArray3DScalarCUDA(array);
        original_value_range.x = output.lower.v;
        original_value_range.y = output.upper.v;
        std::cout << "[Shadow Map] volume range = " << original_value_range.x << " " << original_value_range.y << std::endl;
        set_value_range(data_value_min, data_value_max);
    }

    void set_transfer_function(Array1DFloat4CUDA c, Array1DScalarCUDA a, vec2f r)
    {
        device.tfn.color = c;
        device.tfn.opacity = a;
        set_value_range(r.x, r.y);
        CUDA_SYNC_CHECK();
    }

    void set_transfer_function(array_1d_float4_t c, array_1d_scalar_t a, vec2f r)
    {
        set_transfer_function(CreateArray1DFloat4CUDA(c), CreateArray1DScalarCUDA(a), r);
    }

    void set_transfer_function(const std::vector<float> &c, const std::vector<float> &o, const vec2f &r)
    {
        tfn_colors_data.resize(c.size() / 3);
        for (int i = 0; i < tfn_colors_data.size(); ++i) {
            tfn_colors_data[i].x = c[3 * i + 0];
            tfn_colors_data[i].y = c[3 * i + 1];
            tfn_colors_data[i].z = c[3 * i + 2];
            tfn_colors_data[i].w = 1.f;
        }

        tfn_alphas_data.resize(o.size() / 2);
        for (int i = 0; i < tfn_alphas_data.size(); ++i) {
            tfn_alphas_data[i] = o[2 * i + 1];
        }

        if (!tfn_colors_data.empty() && !tfn_alphas_data.empty()) {
            set_transfer_function(CreateArray1DFloat4CUDA(tfn_colors_data), CreateArray1DScalarCUDA(tfn_alphas_data), r);
        }
        CUDA_SYNC_CHECK();
    }

    void set_value_range(float data_value_min, float data_value_max)
    {
        Array3DScalarCUDA &volume = device.volume;
        if (data_value_max >= data_value_min)
        {
            float normalized_max = integer_normalize(data_value_max, volume.type);
            float normalized_min = integer_normalize(data_value_min, volume.type);
            volume.upper.v = normalized_max; // should use the transfer function value range here
            volume.lower.v = normalized_min;
        }
        volume.scale.v = 1.f / (volume.upper.v - volume.lower.v);
        // Need calculation on max opacity
        auto r_x = max(original_value_range.x, volume.lower.v);
        auto r_y = min(original_value_range.y, volume.upper.v);
        device.tfn.value_range.y = r_y;
        device.tfn.value_range.x = r_x;
        device.tfn.range_rcp_norm = 1.f / (device.tfn.value_range.y - device.tfn.value_range.x);
    }

    void load_lights(Scene &scene)
    {
        lights.clear();
        for (auto& li : scene.lights) {
            if (li.type == scene::Light::DIRECTIONAL) {
                lights.emplace_back(Light{
                    /*.direction =*/ li.directional.direction,
                    /*.intensity =*/ li.intensity
                });
            }
        }

        lights_buffer.alloc_and_upload(lights);

        device.n_lights = (int)lights.size();
        device.lights  = (Light *)lights_buffer.d_pointer();
    }

};

void commit(Scene &scene, HostVolume &volume)
{
    auto &st = scene.instances[0].models[0].volume_model.transfer_function;
    auto &sv = scene.instances[0].models[0].volume_model.volume.structured_regular;

    vec3f scale     = sv.grid_spacing * vec3f(sv.data->dims);
    vec3f translate = sv.grid_origin;

    volume.matrix = affine3f::translate(translate) * affine3f::scale(scale);
    volume.device.transform = volume.matrix;

    volume.load_from_array3d_scalar(sv.data);
    volume.set_transfer_function(CreateArray1DFloat4CUDA(st.color), CreateArray1DScalarCUDA(st.opacity), st.value_range);

    volume.device.step = 1.f / scene.volume_sampling_rate;
    volume.device.step_rcp = scene.volume_sampling_rate;

    volume.device.ao_samples = scene.ao_samples;

    volume.load_lights(scene);

    // call this in the end
    volume.device_buffer.resize(sizeof(volume.device));
    volume.device_buffer.upload(&volume.device, 1);
}

static __device__ bool
intersect_box(float &_t0, float &_t1, const vec3f ray_ori, const vec3f ray_dir, vec3f &box_lower, vec3f &box_upper)
{
    const vec3f &lower = box_lower;
    const vec3f &upper = box_upper;

    float t0 = _t0;
    float t1 = _t1;
#if 1
    const vec3i is_small =
        vec3i(fabs(ray_dir.x) < float_small, fabs(ray_dir.y) < float_small, fabs(ray_dir.z) < float_small);
    const vec3f rcp_dir = /* ray direction reciprocal*/ 1.f / ray_dir;
    const vec3f t_lo = vec3f(is_small.x ? float_large : (lower.x - ray_ori.x) * rcp_dir.x, //
                             is_small.y ? float_large : (lower.y - ray_ori.y) * rcp_dir.y, //
                             is_small.z ? float_large : (lower.z - ray_ori.z) * rcp_dir.z  //
    );
    const vec3f t_hi = vec3f(is_small.x ? -float_large : (upper.x - ray_ori.x) * rcp_dir.x, //
                             is_small.y ? -float_large : (upper.y - ray_ori.y) * rcp_dir.y, //
                             is_small.z ? -float_large : (upper.z - ray_ori.z) * rcp_dir.z  //
    );
    t0 = max(t0, reduce_max(min(t_lo, t_hi)));
    t1 = min(t1, reduce_min(max(t_lo, t_hi)));
#else
    const vec3f t_lo = (lower - ray_ori) / ray_dir;
    const vec3f t_hi = (upper - ray_ori) / ray_dir;
    t0 = max(t0, reduce_max(min(t_lo, t_hi)));
    t1 = min(t1, reduce_min(max(t_lo, t_hi)));
#endif
    _t0 = t0;
    _t1 = t1;
    return t1 > t0;
}

template <typename T>
__forceinline__ __device__ T lerp(float r, const T &a, const T &b)
{
    return (1 - r) * a + r * b;
}

template <typename T, int N>
static __device__ T
array1d_nodal(const ArrayCUDA<1, N> &array, float v)
{
    float t = (0.5f + v * (array.dims.v - 1)) / array.dims.v;
    return tex1D<T>(array.data, t);
}

static __device__ float
sample_volume(const Array3DScalarCUDA &self, vec3f p)
{
    /* sample volume in object space [0, 1] */
    p.x = clamp(p.x, 0.f, 1.f);
    p.y = clamp(p.y, 0.f, 1.f);
    return tex3D<float>(self.data, p.x, p.y, p.z);
}

static __device__ void
sample_transfer_function(const DeviceTransferFunction &tfn, float sampleValue, vec3f &_sampleColor, float &_sampleAlpha)
{
    const auto v = (clamp(sampleValue, tfn.value_range.x, tfn.value_range.y) - tfn.value_range.x) * tfn.range_rcp_norm;
    vec4f rgba = array1d_nodal<float4>(tfn.color, v);
    rgba.w = array1d_nodal<float>(tfn.opacity, v); // followed by the alpha correction
    _sampleColor = vec3f(rgba);
    _sampleAlpha = rgba.w;
}

static __device__ void
opacity_correction(const DeviceVolume &self, const float &distance, float &opacity)
{
    opacity = 1.f - __powf(1.f - opacity, 2.f * self.step_rcp * distance);
}

template <typename F>
__device__ void
ray_marching_iterator(const float tMin, const float tMax,
                      const float step, const F &body,
                      bool debug = false)
{
    vec2f t = vec2f(tMin, min(tMax, tMin + step));
    while ((t.y > t.x) && body(t))
    {
        t.x = t.y;
        t.y = min(t.x + step, tMax);
    }
}

__device__ float
ray_marching_transmittance(const DeviceVolume &self,
                           const vec3f org,
                           const vec3f dir,
                           RandomTEA &rng)
{
    const auto marching_step = self.step;

    float alpha = 0.f;
    float t0 = 0.f, t1 = inf;

    vec3f lower_end = vec3f(0.f);
    vec3f upper_end = vec3f(1.f);

    if (!intersect_box(t0, t1, org, dir, lower_end, upper_end)) return 1.f;

    // jitter ray to remove ringing effects
    const float jitter = rng.get_floats().x;

    // start marching
    ray_marching_iterator(t0, t1, marching_step, [&](const vec2f &t) {
        // sample data value
        const auto p = org + lerp(jitter, t.x, t.y) * dir; // object space position
        const auto sampleValue = sample_volume(self.volume, p);
        // classification
        vec3f sampleColor;
        float sampleAlpha;
        sample_transfer_function(self.tfn, sampleValue, sampleColor, sampleAlpha);
        opacity_correction(self, t.y - t.x, sampleAlpha);
        // blending
        alpha += (1.f - alpha) * sampleAlpha;
        return alpha < nearly_one; 
    });

    return 1.f - alpha;
}

__global__ void
ray_marching_kernel(const vec3i dims, const void *ptr, float *__restrict__ shadowbuffer)
{
    // 3D kernel launch
    vec3i  voxel_coord = vec3i(threadIdx.x + blockIdx.x * blockDim.x, threadIdx.y + blockIdx.y * blockDim.y, threadIdx.z + blockIdx.z * blockDim.z);
    size_t voxel_index = voxel_coord.x + voxel_coord.y * (size_t)dims.x + voxel_coord.z * (size_t)dims.y * (size_t)dims.x;

    if (voxel_index > dims.long_product()) return;

    // generate ray & payload
    RandomTEA rng(voxel_index, 0);

    // voxel center in local coordinate (0-1)^3 get the object to world transformation
    const DeviceVolume &self = *((DeviceVolume *)ptr);

    const affine3f otw = self.transform;
    const affine3f wto = otw.inverse();
    
    const vec3f org = (vec3f(voxel_coord) + vec3f(0.5f, 0.5f, 0.5f)) / vec3f(dims); // transform to object space

    float shadow = 0.f;
    for (int i = 0; i < self.n_lights; ++i)
    {
        auto li = self.lights[i]; // copy to register, intentional

        vec3f li_dir = normalize(li.direction);
        float li_val = li.intensity;

        float li_shadow = 0.f;
        for (int spv = 0; spv < self.ao_samples; ++spv)
        {
            li_shadow += ray_marching_transmittance(self, org, xfmVector(wto, li_dir), rng); // transform to object space
        }

        shadow += (li_shadow / self.ao_samples) * li_val;
    }

    shadowbuffer[voxel_index] = shadow;
}

NAMESPACE_END

using namespace ovr;
using namespace ovr::nnvolume;

struct CmdArgs : CmdArgsBase {
public:
    args::ArgumentParser parser;
    args::HelpFlag help;

    args::Positional<std::string> m_scene;
    std::string scene() { return args::get(m_scene); }

    // optional

    args::ValueFlag<float> m_sampling_rate;
    float sampling_rate() { return (m_sampling_rate) ? args::get(m_sampling_rate) : 1.f; }

    args::ValueFlag<int> m_shadow_samples;
    int shadow_samples() { return (m_shadow_samples) ? args::get(m_shadow_samples) : 1; }

    args::ValueFlag<std::string> m_output;
    std::string output() { return (m_output) ? args::get(m_output) : "shadowmap"; }

    // group for random lights

    args::Group group_random_lights;

    args::ValueFlag<int> m_random_lights;
    bool random_lights() { return (m_random_lights); }
    int num_random_lights() { return (m_random_lights) ?  args::get(m_random_lights) : 0; }

    args::Flag m_random_tfn;
    bool random_tfn() { return (m_random_tfn); }

    // Ring light parsing

    args::Group group_ring_lights;

    args::ValueFlag<int> m_ring_lights;
    bool ring_lights() { return (m_ring_lights); }
    int num_ring_lights() { return (m_ring_lights) ?  args::get(m_ring_lights) : 0; }

    args::ValueFlag<float> m_theta;
    float value_theta() { return (m_theta) ? args::get(m_theta) : 0.f; }

    args::ValueFlag<float> m_phi;
    float value_phi() { return (m_phi) ? args::get(m_phi) : 0.f; }

public:
    CmdArgs(const char *title, int argc, char **argv)
        : parser(title)
        , help(parser, "help", "display the help menu", {'h', "help"})
        , m_scene(parser, "string", "the scene to render")
        , m_sampling_rate(parser, "float", "ray marching sampling rate", {"sampling-rate"})
        , m_shadow_samples(parser, "int", "number of samples per voxel", {"shadow-samples"})
        , m_output(parser, "string", "output name", {"output"})
        , group_random_lights(parser, "random light group:", args::Group::Validators::AllOrNone)
        , m_random_lights(group_random_lights, "int", "generate N random lights", {"random-lights"})
        , m_random_tfn(parser, "flag", "generate a random transfer function", {"random-tfn"})
        , group_ring_lights(parser, "ring light group:", args::Group::Validators::AllOrNone)
        , m_ring_lights(group_ring_lights, "int", "Generate N lights in a ring", {"ring-lights"})
        , m_theta(parser, "int", "ring theta", {"theta"})
        , m_phi(parser, "int", "ring phi", {"phi"})
    {
        exec(parser, argc, argv);
    }
};

int main(int ac, char **av)
{
    CmdArgs args("Shadow Volume Generator", ac, av);

    // Create scene + volume + tfn + lights
    Scene scene = scene::create_scene(args.scene());
    scene.volume_sampling_rate = args.sampling_rate();
    scene.ao_samples = args.shadow_samples();

    auto &scene_vol = scene.instances[0].models[0].volume_model.volume.structured_regular;
    auto &scene_tfn = scene.instances[0].models[0].volume_model.transfer_function;

    // generate random lights
    srand((unsigned int)time(NULL)); // Initialization, should only be called once.

    if (args.random_lights()) {

        scene.lights.clear();

        for (int i = 0; i < args.num_random_lights(); ++i) {
            std::cout << "Generate Light #" << i << std::endl;

            scene::Light light;
            light.type = scene::Light::DIRECTIONAL;

            float theta = 2.0f * (float)M_PI * ((float)rand() / (float)(RAND_MAX));
            float phi   = 1.0f * (float)M_PI * ((float)rand() / (float)(RAND_MAX));

            // Generate Direction
            float x = 1.0f * cos(phi) * sin(theta);
            float y = 1.0f * sin(phi) * sin(theta);
            float z = 1.0f * cos(theta);
            light.directional.direction = normalize(vec3f(x, y, z));
            std::cout << "Light Direction: " << light.directional.direction.x << " " << light.directional.direction.y << " " << light.directional.direction.z << " " << std::endl;

            // Generate Color
            light.intensity = 1.f / args.num_random_lights();

            // Store
            scene.lights.push_back(light);
        }

    }

    // Ring light generation
    else if (args.ring_lights()){
        std::cout << "Num Lights: " << args.num_ring_lights() << std::endl;
        std::cout << "Theta: " << args.value_theta() << std::endl;
        std::cout << "Phi: " << args.value_phi() << std::endl << std::endl;

        scene.lights.clear();

        for (int i = 0; i < args.num_ring_lights(); ++i) {
            std::cout << "Generate Light #" << i << std::endl;

            scene::Light light;
            light.type = scene::Light::DIRECTIONAL;

            // Theta and phi are passed in as deg
            float theta = (float)M_PI/180.f * args.value_theta();

            float phi_offset = 360.f / args.num_ring_lights();
            float phi   = (float)M_PI/180.f * (args.value_phi() + i * phi_offset);

            // Generate Direction
            float x = 1.0f * cos(phi) * sin(theta);
            float y = 1.0f * sin(phi) * sin(theta);
            float z = 1.0f * cos(theta);
            light.directional.direction = normalize(vec3f(x, y, z));
            std::cout << "Light Direction: " << light.directional.direction.x << " " << light.directional.direction.y << " " << light.directional.direction.z << " " << std::endl;

            // Generate Color
            light.intensity = 1.f / args.num_ring_lights();

            // Store
            scene.lights.push_back(light);
        }

    }


    // create a transfer function object
    tfn::TransferFunctionCore tfn(1024);
    range1f range;
    {
        vec4f* color_data = scene_tfn.color->data_typed<vec4f>();
        float* alpha_data = scene_tfn.opacity->data_typed<float>();
        for (int i = 0; i < scene_tfn.color->size(); ++i) {
            auto color = color_data[i];
            float pos = (float)i / (scene_tfn.color->size() - 1);
            tfn.addColorControl(tfn::TransferFunctionCore::ColorControl(pos, color.xyz()));
        }
        for (int i = 0; i < scene_tfn.color->size(); ++i) {
            auto alpha = alpha_data[i];
            float pos = (float)i / (scene_tfn.color->size() - 1);
            tfn.addAlphaControl(vec2f(pos, alpha));
        }

        range.lower = scene_tfn.value_range.x;
        range.upper = scene_tfn.value_range.y;
    }

    if (args.random_tfn()) {
        tfn.clearAlphaControls();

        // Create Number of Gaussian, 1 to 10
        const int num_gaussian = (int)(((float)rand() / (float)(RAND_MAX) + 1) * 5);
        std::cout << "Generate Gaussian #" << num_gaussian << std::endl;

        for (int each_gaussian = 0; each_gaussian < num_gaussian; ++each_gaussian)
        {
            float gaussian_mean   = (float)rand() / (float)(RAND_MAX);
            float gaussian_sigma  = max(0.2f * (float)rand() / (float)(RAND_MAX), 0.0001f);
            float gaussian_height = max((gaussian_sigma * std::sqrt(2.0f * float(M_PI))) * (float)rand() / (float)(RAND_MAX), 0.0001f);

            std::cout << "Generate Gaussian Mean: " << gaussian_mean << std::endl;
            std::cout << "Generate Gaussian Height: " << gaussian_height << std::endl;
            std::cout << "Generate Gaussian Sigma: " << gaussian_sigma << std::endl;

            tfn.addGaussianObject(gaussian_mean, gaussian_sigma, gaussian_height);
        }

        // It seems we do not need to explicitly normalize gaussians
    }

    // overwrite the scene tfn
    tfn.updateColorMap();
    {
        auto* table = (vec4f*)tfn.data();
        std::vector<vec4f> color(tfn.resolution());
        std::vector<float> alpha(tfn.resolution());
        for (int i = 0; i < tfn.resolution(); ++i) {
            const auto rgba = table[i];
            color[i] = vec4f(rgba.xyz(), 1.f);
            alpha[i] = rgba.w;
        }
        scene_tfn.color   = CreateArray1DFloat4(color);
        scene_tfn.opacity = CreateArray1DScalar(alpha);
    }

    // set volume
    HostVolume params;
    commit(scene, params);

    // process
    vec3i  shadowmap_dims = vec3i(256); // vec3i(scene_vol.data->dims);
    size_t shadowmap_size = shadowmap_dims.long_product();
    CUDABuffer         shadowmap_gpu;
    std::vector<float> shadowmap_cpu;
    range1f shadowmap_range;

    shadowmap_cpu.resize(shadowmap_size);
    shadowmap_gpu.alloc(shadowmap_size * sizeof(float));

    CUDA_SYNC_CHECK();

    // call kernel to compute shadow volume
    const int n_threads = 8;
    const dim3 block_size(n_threads, n_threads, n_threads);
    const dim3 grid_size(
        misc::div_round_up(shadowmap_dims.x, n_threads),
        misc::div_round_up(shadowmap_dims.y, n_threads),
        misc::div_round_up(shadowmap_dims.z, n_threads)
    );
    ray_marching_kernel<<<grid_size, block_size>>>(shadowmap_dims, (void *)params.device_buffer.d_pointer(), (float *)shadowmap_gpu.d_pointer());

    CUDA_SYNC_CHECK();

    // shadowmap_gpu.download_async(shadowmap_cpu.data(), shadowmap_cpu.size());
    cudaMemcpy(shadowmap_cpu.data(), (float*)shadowmap_gpu.d_pointer(), shadowmap_size * sizeof(float), cudaMemcpyDeviceToHost);

    CUDA_SYNC_CHECK();

    for (int i = 0; i < shadowmap_size; i++) {
        shadowmap_range.extend(shadowmap_cpu[i]);
    }
    std::cout << "shadowmap range: " << shadowmap_range.lower << " " << shadowmap_range.upper << std::endl;

    // save shadow volume to a binary file
    std::ofstream outS(args.output() + ".bin", std::ios::out | std::ios::binary);
    outS.write((char *)shadowmap_cpu.data(), shadowmap_cpu.size() * sizeof(float)); // <- This is where the code breaks
    outS.close();

    // save the scene file
    json root;

    // volume data
    {
        json data;
        data["dimensions"] = { { "x", shadowmap_dims.x }, { "y", shadowmap_dims.y }, { "z", shadowmap_dims.z } };
        data["endian"] = "LITTLE_ENDIAN";
        data["fileName"] = args.output() + ".bin";
        data["fileUpperLeft"] = false;
        data["format"] = "REGULAR_GRID_RAW_BINARY";
        data["id"] = 1;
        data["name"] = "shadowmap";
        data["offset"] = 0;
        data["type"] = "FLOAT";
        root["dataSource"].push_back(data);
    }

    root["original"] = args.scene();

    // view
    {
        json& view = root["view"];

        json& camera = view["camera"];
        camera["center"] = { { "x", shadowmap_dims.x / 2.f }, { "y", shadowmap_dims.y / 2.f }, { "z", shadowmap_dims.z / 2.f } };
        camera["eye"]    = { { "x", shadowmap_dims.x / 2.f }, { "y", shadowmap_dims.y / 2.f }, { "z", shadowmap_dims.z / 2.f - shadowmap_dims.z } };
        camera["up"]     = { { "x", 0.f }, { "y", 1.f }, { "z", 0.f } };
        camera["fovy"] = 60;
        camera["projectionMode"] = "PERSPECTIVE";
        camera["zFar"] = 2000;
        camera["zNear"] = 1;

        for (auto& li : params.lights) {
            json light;
            light["ambient"]  = { { "a", 1.f }, { "b", 1.f }, { "g", 1.f }, { "r", 1.f } };
            light["specular"] = { { "a", 1.f }, { "b", 1.f }, { "g", 1.f }, { "r", 1.f } };
            light["diffuse"]  = { { "a", 1.f }, { "r", li.intensity   }, { "g", li.intensity   }, { "b", li.intensity   } };
            light["position"] = { { "w", 0.f }, { "x", li.direction.x }, { "y", li.direction.y }, { "z", li.direction.z } };
            light["type"] = "DIRECTIONAL_LIGHT";
            if (!view.contains("lightSource")) {
                view["lightSource"] = light;
            }
            else {
                view["additionalLightSources"].push_back(light);
            }
        }

        view["lighting"] = true;
        view["lightingSide"] = "FRONT_SIDE";
        view["tfPreIntegration"] = false;

        auto& vol = view["volume"];
        vol["dataId"] = 1;
        vol["interpolationType"] = "LINEAR_INTERPOLATION";
        vol["opacityUnitDistance"] = 1;
        vol["sampleDistance"] = params.device.step;
        vol["scalarMappingRange"] = {
            { "maximum", 1.0 }, { "minimum", 0.0 } // we should not normalize a shadow map
        };
        vol["transferFunctionType"] =  "TRANSFER_FUNCTION";
        vol["visible"] = true;

        // transfer function
        tfn::saveTransferFunction(tfn, vol["transferFunction"]);
    }

    // save as text file
    std::ofstream outJ(args.output() + ".json", std::ios::out);
    outJ << std::setw(4) << root << std::endl;
    outJ.close();

    std::cout << "Ended" << std::endl;
    return 0;
}

// command to train a neural network
// ../../instant-vnr-cuda/run.sh ../../instant-vnr-cuda/build/Release/vnr_cmd_train --volume ./shadowmap.json   --max-num-steps 10000 --mode GPU --network network.json 

// command to run
// bash ../scripts/run.sh ../build/Debug/renderapp configs/scene_mechhand.json nnvolume
