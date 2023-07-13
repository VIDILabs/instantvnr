#pragma once
#ifdef ENABLE_IN_SHADER

#include <api.h>

void vnrMarchingCube(vnrVolume volume, float isovalue, vnr::vec3f** ptr, size_t* size, bool cuda);

void vnrSaveTriangles(std::string filename, const vnr::vec3f* ptr, size_t size);

#endif
