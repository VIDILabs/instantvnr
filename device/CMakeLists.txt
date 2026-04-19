################
# device_nnvolume_array — OBJECT library used by the shadow-map app and as a
# build unit for the device plugin.  Standalone builds still compile it so the
# full feature set is preserved; it is only *linked* when OVR_BUILD_MODULE_NNVOLUME
# is defined (i.e., when building as part of the OVR plugin framework).
################
add_library(device_nnvolume_array OBJECT
  device_nnvolume_array.cpp
)
set_target_properties(device_nnvolume_array
PROPERTIES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE TRUE
)
target_link_libraries(device_nnvolume_array PRIVATE rendercommon TBB::tbb)

################
# device_nnvolume — standalone build uses SHARED so the library is installed
# alongside other .so files.  In OVR plugin mode the original MODULE type is
# appropriate; use the INSTANTVNR_STANDALONE cache variable to switch.
################
option(INSTANTVNR_STANDALONE "Build device_nnvolume as SHARED (pip package) instead of MODULE (OVR plugin)" ON)

if(INSTANTVNR_STANDALONE)
  set(_dev_type SHARED)
else()
  set(_dev_type MODULE)
endif()

add_library(device_nnvolume ${_dev_type}
  $<TARGET_OBJECTS:device_nnvolume_array>
  device.cpp
  device_impl.cpp
  # method_shadowmap.cu  # uncomment to include shadow-map GPU code
)
set_target_properties(device_nnvolume
PROPERTIES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries     (device_nnvolume PRIVATE rendercommon TBB::tbb instantvnr)
target_compile_options    (device_nnvolume PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_compile_definitions(device_nnvolume PRIVATE
  ${IVNR_GLIBCXX_CXX11_ABI}
  MACROCELL_SIZE_MIP=${MACROCELL_SIZE_MIP}
)
