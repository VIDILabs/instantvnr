################
#
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
#
################
add_library(device_nnvolume MODULE
  $<TARGET_OBJECTS:device_nnvolume_array>
  device.cpp
  device_impl.cpp
  # method_shadowmap.cu
)
set_target_properties(device_nnvolume 
PROPERTIES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_SEPARABLE_COMPILATION ON
)

################
#
################
target_link_libraries     (device_nnvolume PRIVATE rendercommon TBB::tbb instantvnr)
target_compile_options    (device_nnvolume PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_compile_definitions(device_nnvolume PRIVATE OVR_BUILD_MODULE_NNVOLUME=1)
target_compile_definitions(device_nnvolume PRIVATE MACROCELL_SIZE_MIP=${MACROCELL_SIZE_MIP})
