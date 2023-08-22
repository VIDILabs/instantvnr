#pragma once
#ifdef ENABLE_IN_SHADER

#include <api.h>

double vnrMarchingCube(vnrVolume volume, float isovalue, vnr::vec3f** ptr, size_t* size, bool cuda);

#endif
