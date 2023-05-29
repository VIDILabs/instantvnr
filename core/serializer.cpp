#include "serializer.h"

namespace tfn {
typedef vnr::math::vec2f vec2f;
typedef vnr::math::vec2i vec2i;
typedef vnr::math::vec3f vec3f;
typedef vnr::math::vec3i vec3i;
typedef vnr::math::vec4f vec4f;
typedef vnr::math::vec4i vec4i;
} // namespace tfn
#define TFN_MODULE_EXTERNAL_VECTOR_TYPES
#include "tfn/core.h"

// JSON I/O
#include <json/json.hpp>

namespace vnr { // clang-format off

struct VolumeDesc : MultiVolume::File 
{
  vec3i dims;
  ValueType type;
};

NLOHMANN_JSON_SERIALIZE_ENUM(ValueType, {
  { ValueType::VALUE_TYPE_INT8, "BYTE" },
  { ValueType::VALUE_TYPE_UINT8, "UNSIGNED_BYTE" },
  { ValueType::VALUE_TYPE_INT16, "SHORT" },
  { ValueType::VALUE_TYPE_UINT16, "UNSIGNED_SHORT" },
  { ValueType::VALUE_TYPE_INT32, "INT" },
  { ValueType::VALUE_TYPE_UINT32, "UNSIGNED_INT" },
  { ValueType::VALUE_TYPE_FLOAT, "FLOAT" },
  { ValueType::VALUE_TYPE_DOUBLE, "DOUBLE" },
}); // clang-format on

enum Endianness { VNR_LITTLE_ENDIAN, VNR_BIG_ENDIAN };
NLOHMANN_JSON_SERIALIZE_ENUM(Endianness, {
  { VNR_LITTLE_ENDIAN, "LITTLE_ENDIAN" },
  { VNR_BIG_ENDIAN, "BIG_ENDIAN" },
}); // clang-format on

#define define_vector_serialization(T)                      \
   NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(vec2##T, x, y);       \
   NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(vec3##T, x, y, z);    \
   NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(vec4##T, x, y, z, w); \
   NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(range1##T, minimum, maximum);
define_vector_serialization(i);
define_vector_serialization(f);
#undef define_vector_serialization

#define assert_throw(x, msg) { if (!(x)) throw std::runtime_error(msg); }

template<typename ScalarT>
inline ScalarT
scalar_from_json(const json& in);

#define define_scalar_serialization(T) template<> inline T scalar_from_json<T>(const json& in) { return in.get<T>(); }
define_scalar_serialization(std::string);
define_scalar_serialization(bool);
define_scalar_serialization(int64_t);
define_scalar_serialization(uint64_t);
define_scalar_serialization(double);

template<typename ScalarT/*, typename std::enable_if_t<!std::is_arithmetic<ScalarT>::value> = true*/>
inline ScalarT
scalar_from_json(const json& in)
{
  ScalarT v;
  from_json(in, v);
  return v;
}

template<typename ScalarT>
inline ScalarT
scalar_from_json(const json& in, const std::string& key)
{
  assert_throw(in.is_object(), "has to be a JSON object");
  assert_throw(in.contains(key), "incorrect key: " + key);
  return scalar_from_json<ScalarT>(in[key]);
}

template<typename ScalarT>
inline ScalarT
scalar_from_json(const json& in, const std::string& key, const ScalarT& value)
{
  assert_throw(in.is_object(), "has to be a JSON object");
  if (in.contains(key)) {
    return scalar_from_json<ScalarT>(in[key]);
  }
  else {
    return value;
  }
}

namespace {

vnr::vec2f
rangeFromJson(vnr::json jsrange)
{
  if (!jsrange.contains("minimum") || 
      !jsrange.contains("maximum")) 
  {
    return vnr::vec2f(0.0, 0.0);
  }
  return vnr::vec2f(jsrange["minimum"].get<float>(), 
                    jsrange["maximum"].get<float>());
}

static bool
file_exists_test(const std::string& name)
{
  std::ifstream f(name.c_str());
  return f.good();
}

static std::string
valid_filename(const json& in, const std::string& key)
{
  if (in.contains(key)) {
    auto& js = in[key];
    if (js.is_array()) {
      for (auto& s : js) {
        if (file_exists_test(s.get<std::string>())) return s.get<std::string>();
      }
      throw std::runtime_error("Cannot find volume file.");
    }
    else {
      return js.get<std::string>();
    }
  }
  else {
    throw std::runtime_error("Json key 'fileName' doesnot exist");
  }
}

} // namespace

void create_json_volume_stringify_diva(json root, MultiVolume& volume)
{
  const auto config = root["volume"];
  const vec2f range = scalar_from_json<vec2f>(config, "range");

  volume.dims = scalar_from_json<vec3i>(config["dims"]);
  volume.type = scalar_from_json<ValueType>(config["type"]);
  volume.range.lower = range.x;
  volume.range.upper = range.y;

  if (config["filename"].is_array()) {
    for (auto& v : config["filename"]) {
      MultiVolume::File f;
      f.filename = v.get<std::string>();
      f.bigendian = config.contains("bigendian") ? config["bigendian"].get<bool>() : false;
      f.offset = 0;
      volume.data.push_back(f);
    }
  }

  else {
    MultiVolume::File file;
    file.filename = config["filename"].get<std::string>();
    file.bigendian = config.contains("bigendian") ? config["bigendian"].get<bool>() : false;
    file.offset = 0;
    volume.data.push_back(file);
  }
  
  volume.scale = vec3f(volume.dims);
  volume.translate = -vec3f(volume.dims) / 2.f;
}

void
create_json_scene_diva(json root, MultiVolume& volume, TransferFunction& tfn, Camera& camera)
{
  create_json_volume_stringify_diva(root, volume);
  // TODO load TFN and Camera //
}

Camera
create_scene_vidi__camera(const json& jscamera)
{
  Camera camera;
  camera.from = scalar_from_json<vec3f>(jscamera["eye"]);
  camera.at = scalar_from_json<vec3f>(jscamera["center"]);
  camera.up = scalar_from_json<vec3f>(jscamera["up"]);
  camera.fovy = jscamera["fovy"].get<float>();

  return camera;
}

TransferFunction
create_scene_vidi__tfn(json jstfn, json jsvolume, ValueType type)
{
  tfn::TransferFunctionCore tf;
  tfn::loadTransferFunction(jstfn, tf);

  /* finalize */
  TransferFunction tfn;

  auto* table = (vec4f*)tf.data();
  std::vector<vec3f> color(tf.resolution());
  std::vector<vec2f> alpha(tf.resolution());
  for (int i = 0; i < tf.resolution(); ++i) {
    auto rgba = table[i];
    color[i] = rgba.xyz();
    alpha[i] = vec2f((float)i / (tf.resolution() - 1), rgba.w);
  }
  if (alpha[0].y < 0.01f) alpha[0].y = 0.f;
  if (alpha[tf.resolution()-1].y < 0.01f) alpha[tf.resolution()-1].y = 0.f;
  tfn.color = std::move(color);
  tfn.alpha = std::move(alpha);

  if (jsvolume.contains("scalarMappingRangeUnnormalized")) {
    auto r = rangeFromJson(jsvolume["scalarMappingRangeUnnormalized"]);
    tfn.range.lower = r.x;
    tfn.range.upper = r.y;
  }

  /* try it ... */
  else if (jsvolume.contains("scalarMappingRange")) {
    auto r = rangeFromJson(jsvolume["scalarMappingRange"]);
    switch (type) {
    case VALUE_TYPE_UINT8:
      tfn.range.lower = std::numeric_limits<uint8_t>::max() * r.x;
      tfn.range.upper = std::numeric_limits<uint8_t>::max() * r.y;
      break;
    case VALUE_TYPE_INT8:
      tfn.range.lower = std::numeric_limits<int8_t>::max() * r.x;
      tfn.range.upper = std::numeric_limits<int8_t>::max() * r.y;
      break;
    case VALUE_TYPE_UINT16:
      tfn.range.lower = std::numeric_limits<uint16_t>::max() * r.x;
      tfn.range.upper = std::numeric_limits<uint16_t>::max() * r.y;
      break;
    case VALUE_TYPE_INT16:
      tfn.range.lower = std::numeric_limits<int16_t>::max() * r.x;
      tfn.range.upper = std::numeric_limits<int16_t>::max() * r.y;
      break;
    case VALUE_TYPE_UINT32:
      tfn.range.lower = std::numeric_limits<uint32_t>::max() * r.x;
      tfn.range.upper = std::numeric_limits<uint32_t>::max() * r.y;
      break;
    case VALUE_TYPE_INT32:
      tfn.range.lower = std::numeric_limits<int32_t>::max() * r.x;
      tfn.range.upper = std::numeric_limits<int32_t>::max() * r.y;
      break;
    case VALUE_TYPE_FLOAT:
    case VALUE_TYPE_DOUBLE:
      tfn.range.lower = r.x;
      tfn.range.upper = r.y;
      break;
    default: throw std::runtime_error("unknown data type");
    }
  }

  else {
    /* calculate the volume value range ... */
    // throw std::runtime_error("unknown data range");
  }

  return tfn;
}

VolumeDesc
create_scene_vidi__volume(const json& jsdata)
{
  VolumeDesc volume;

  const auto format = scalar_from_json<std::string>(jsdata["format"]);

  if (format == "REGULAR_GRID_RAW_BINARY") {
    const auto filename      = valid_filename(jsdata, "fileName");
    const auto dims          = scalar_from_json<vec3i>(jsdata, "dimensions");
    const auto type          = scalar_from_json<ValueType>(jsdata, "type");
    const auto offset        = scalar_from_json<size_t>(jsdata, "offset", 0);
    const auto flipped       = scalar_from_json<bool>(jsdata, "fileUpperLeft", false);
    const auto is_big_endian = scalar_from_json<Endianness>(jsdata, "endian", VNR_LITTLE_ENDIAN) == VNR_BIG_ENDIAN;
    volume.filename = filename;
    volume.offset = offset;
    volume.bigendian = is_big_endian;
    volume.dims = dims;
    volume.type = type;
  }
  else {
    throw std::runtime_error("data type unimplemented");
  }

  return volume;
}

MultiVolume::File
create_scene_vidi__multivolume(const json& jsdata, const VolumeDesc& volume)
{
  MultiVolume::File file;

  const auto format = scalar_from_json<std::string>(jsdata["format"]);

  if (format == "REGULAR_GRID_RAW_BINARY") {
    const auto filename      = valid_filename(jsdata, "fileName");
    const auto dims          = scalar_from_json<vec3i>(jsdata, "dimensions");
    const auto type          = scalar_from_json<ValueType>(jsdata, "type");
    const auto offset        = scalar_from_json<size_t>(jsdata, "offset", 0);
    const auto flipped       = scalar_from_json<bool>(jsdata, "fileUpperLeft", false);
    const auto is_big_endian = scalar_from_json<Endianness>(jsdata, "endian", VNR_LITTLE_ENDIAN) == VNR_BIG_ENDIAN;
    assert(volume.type == type);
    assert(volume.dims == dims);
    file.filename = filename;
    file.offset = offset;
    file.bigendian = is_big_endian;
  }
  else {
    throw std::runtime_error("data type unimplemented");
  }

  return file;
}

ValueType
create_scene_vidi__datatype(const json& jsdata)
{
  const auto format = scalar_from_json<std::string>(jsdata["format"]);
  if (format == "REGULAR_GRID_RAW_BINARY") {
    return scalar_from_json<ValueType>(jsdata, "type");
  }
  else {
    throw std::runtime_error("data type unimplemented");
  }
}

void
create_json_scene_vidi(json root, MultiVolume& volume, TransferFunction& tfn, Camera& camera)
{
  const auto& ds = root["dataSource"];
  assert_throw(ds.is_array(), "'dataSource' is expected to be an array");

  // reate primary volume
  VolumeDesc pv = create_scene_vidi__volume(ds[0]);

  // create multi volume
  volume.dims = pv.dims;
  volume.type = pv.type;
  volume.data.resize(ds.size()); 
  volume.data[0] = pv; // slicing
  for (int i = 1; i < ds.size(); ++i) {
    volume.data.push_back(create_scene_vidi__multivolume(ds[i], pv));
  }

  // load the transfer function as well as the value range
  tfn = create_scene_vidi__tfn(root["view"]["volume"]["transferFunction"], root["view"]["volume"], pv.type);
  volume.range = tfn.range;

  // when an integer volume is being used, the data value will be normalized, if an unnormalized value range is not present, we produce a warning
  if (!root["view"]["volume"].contains("scalarMappingRangeUnnormalized")) {
    auto type = scalar_from_json<ValueType>(ds[0]["type"]);
    if (type != VALUE_TYPE_FLOAT && type != VALUE_TYPE_DOUBLE) {
      std::cerr << "[vidi] An unnormalized value range cannot be found for transfer function, incorrect results can be produced." << std::endl;
    }
  }

  // camera
  camera = create_scene_vidi__camera(root["view"]["camera"]);
  camera.at   -= vec3f(volume.dims) / 2.f;
  camera.from -= vec3f(volume.dims) / 2.f;
}

// void
// create_json_data_type_stringify_vidi(json root, ValueType& type)
// {
//   const auto& ds = root["dataSource"];
//   assert_throw(ds.is_array(), "'dataSource' is expected to be an array");
//   assert_throw(ds.size() >= 1, "'dataSource' should contain at least one element");
// 
//   type = create_scene_vidi__datatype(ds[0]);
// }

void
create_json_tfn_stringify_vidi(json root, TransferFunction& tfn)
{
  ValueType type; // create_json_data_type_stringify_vidi(root, type);

  const auto& ds = root["dataSource"];
  assert_throw(ds.is_array(), "'dataSource' is expected to be an array");
  assert_throw(ds.size() >= 1, "'dataSource' should contain at least one element");

  type = create_scene_vidi__datatype(ds[0]);

  // load the transfer function as well as the value range
  tfn = create_scene_vidi__tfn(root["view"]["volume"]["transferFunction"], root["view"]["volume"], type);
}

void
create_json_volume_stringify_vidi(json root, MultiVolume& volume)
{
  const auto& ds = root["dataSource"];
  assert_throw(ds.is_array(), "'dataSource' is expected to be an array");

  // construct file descriptors
  VolumeDesc pv = create_scene_vidi__volume(ds[0]);

  // create multi volume
  volume.dims = pv.dims;
  volume.type = pv.type;
  volume.data.resize(ds.size()); 
  volume.data[0] = pv; // slicing
  for (int i = 1; i < ds.size(); ++i) {
    volume.data.push_back(create_scene_vidi__multivolume(ds[i], pv));
  }

  // load the transfer function as well as the value range
  TransferFunction tfn;
  tfn = create_scene_vidi__tfn(root["view"]["volume"]["transferFunction"], root["view"]["volume"], pv.type);
  volume.range = tfn.range;
}

void
create_json_camera_stringify_vidi(json root, Camera& camera)
{
  camera = create_scene_vidi__camera(root["view"]["camera"]);

  const auto& ds = root["dataSource"];
  assert_throw(ds.is_array(), "'dataSource' is expected to be an array");
  VolumeDesc pv = create_scene_vidi__volume(ds[0]);

  camera.at   -= vec3f(pv.dims) / 2.f;
  camera.from -= vec3f(pv.dims) / 2.f;
}

void
create_json_scene_stringify(json root, MultiVolume& volume, TransferFunction& tfn, Camera& camera)
{
  assert(root.is_object());

  if (root.contains("version")) {
    if (root["version"] == "DIVA") {
      return create_json_scene_diva(root, volume, tfn, camera);
    }
    else if (root["version"] == "VIDI3D") {
      return create_json_scene_vidi(root, volume, tfn, camera);
    }
    else throw std::runtime_error("unknown JSON configuration format");
  }
  return create_json_scene_vidi(root, volume, tfn, camera);
}

void
create_json_volume_stringify(json root, MultiVolume& volume)
{
  assert(root.is_object());
  if (root.contains("version")) {
    if      (root["version"] == "DIVA"  ) return create_json_volume_stringify_diva(root, volume);
    else if (root["version"] == "VIDI3D") {}
    else throw std::runtime_error("unknown JSON configuration format");
  }
  return create_json_volume_stringify_vidi(root, volume);
}

void
create_json_tfn_stringify(json root, TransferFunction& tfn)
{
  assert(root.is_object());
  if (root.contains("version")) {
    if      (root["version"] == "DIVA"  ) return; // TODO 
    else if (root["version"] == "VIDI3D") {}
    else throw std::runtime_error("unknown JSON configuration format");
  }
  return create_json_tfn_stringify_vidi(root, tfn);
}

void
create_json_camera_stringify(json root, Camera& camera)
{
  assert(root.is_object());
  if (root.contains("version")) {
    if      (root["version"] == "DIVA"  ) return; // TODO 
    else if (root["version"] == "VIDI3D") {}
    else throw std::runtime_error("unknown JSON configuration format");
  }
  return create_json_camera_stringify_vidi(root, camera);
}

} // namespace ovr
