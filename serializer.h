#pragma once

#include "core/instantvnr_types.h"

#include <json/json.hpp>

#include <fstream>
#include <string>

namespace vnr {

using json = nlohmann::json;

void
create_json_scene_stringify(json root, MultiVolume& volume, TransferFunction& tfn, Camera& camera);

inline void
create_json_scene(std::string filename, MultiVolume& volume, TransferFunction& tfn, Camera& camera)
{
  std::ifstream file(filename);
  json root = json::parse(file, nullptr, true, true);
  return create_json_scene_stringify(root, volume, tfn, camera);
}

void
create_json_tfn_stringify(json root, TransferFunction& tfn);

inline void
create_json_tfn(std::string filename, TransferFunction& tfn)
{
  std::ifstream file(filename);
  json root = json::parse(file, nullptr, true, true);
  return create_json_tfn_stringify(root, tfn);
}

void
create_json_volume_stringify(json root, MultiVolume& volume);

inline void
create_json_volume(std::string filename, MultiVolume& volume)
{
  std::ifstream file(filename);
  json root = json::parse(file, nullptr, true, true);
  return create_json_volume_stringify(root, volume);
}

void
create_json_camera_stringify(json root, Camera& camera);

inline void
create_json_camera(std::string filename, Camera& camera)
{
  std::ifstream file(filename);
  json root = json::parse(file, nullptr, true, true);
  return create_json_camera_stringify(root, camera);
}

} // namespace ovr
