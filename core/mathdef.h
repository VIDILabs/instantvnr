//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //

/**
 * Geometry Types Defined by the Application
 */
#ifndef OVR_MATHDEF_H
#define OVR_MATHDEF_H

#include <gdt/math/mat.h>
#include <gdt/math/vec.h>
#include <gdt/math/box.h>

namespace vnr {

// ------------------------------------------------------------------
// Math Functions
// ------------------------------------------------------------------

namespace math = gdt;
using vec2f = math::vec2f;
using vec2i = math::vec2i;
using vec3f = math::vec3f;
using vec3i = math::vec3i;
using vec4f = math::vec4f;
using vec4i = math::vec4i;
using range1i = math::range1i;
using range1f = math::range1f;
using box3i = math::box3i;
using box3f = math::box3f;
using affine3f = math::affine3f;
using linear3f = math::linear3f;
using math::clamp;
using math::max;
using math::min;
using math::floor;
using math::ceil;
using math::xfmNormal;
using math::xfmPoint;
using math::xfmVector;

// ------------------------------------------------------------------
// Scalar Definitions
// ------------------------------------------------------------------

enum ValueType {
  VALUE_TYPE_UINT8,
  VALUE_TYPE_INT8,
  VALUE_TYPE_UINT16,
  VALUE_TYPE_INT16,
  VALUE_TYPE_UINT32,
  VALUE_TYPE_INT32,
  VALUE_TYPE_UINT64,
  VALUE_TYPE_INT64,
  VALUE_TYPE_FLOAT,
  VALUE_TYPE_FLOAT2,
  VALUE_TYPE_FLOAT3,
  VALUE_TYPE_FLOAT4,
  VALUE_TYPE_DOUBLE,
};

inline __both__ int
value_type_size(ValueType type)
{
  switch (type) {
  case VALUE_TYPE_UINT8:
  case VALUE_TYPE_INT8: return sizeof(char);
  case VALUE_TYPE_UINT16:
  case VALUE_TYPE_INT16: return sizeof(short);
  case VALUE_TYPE_UINT32:
  case VALUE_TYPE_INT32: return sizeof(int);
  case VALUE_TYPE_UINT64:
  case VALUE_TYPE_INT64: return sizeof(int64_t);
  case VALUE_TYPE_FLOAT: return sizeof(float);
  case VALUE_TYPE_DOUBLE: return sizeof(double);
  default: return 0;
  }
}

template<typename T> __both__ ValueType value_type();
template<> inline __both__ ValueType value_type<uint8_t >() { return VALUE_TYPE_UINT8;  }
template<> inline __both__ ValueType value_type<int8_t  >() { return VALUE_TYPE_INT8;   }
template<> inline __both__ ValueType value_type<uint16_t>() { return VALUE_TYPE_UINT16; }
template<> inline __both__ ValueType value_type<int16_t >() { return VALUE_TYPE_INT16;  }
template<> inline __both__ ValueType value_type<uint32_t>() { return VALUE_TYPE_UINT32; }
template<> inline __both__ ValueType value_type<int32_t >() { return VALUE_TYPE_INT32;  }
template<> inline __both__ ValueType value_type<uint64_t>() { return VALUE_TYPE_UINT64; }
template<> inline __both__ ValueType value_type<int64_t >() { return VALUE_TYPE_INT64;  }
template<> inline __both__ ValueType value_type<float   >() { return VALUE_TYPE_FLOAT;  }
template<> inline __both__ ValueType value_type<double  >() { return VALUE_TYPE_DOUBLE; }

}


#endif//OVR_MATHDEF_H
