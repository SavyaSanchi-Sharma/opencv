# Detect and configure AMD ROCm/HIP support.
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# Author: Jeff Daily <jeff.daily@amd.com>

if(NOT CMAKE_HIP_COMPILER)
  message(STATUS "HIP: hipcc not enabled as a language, HIP support disabled.")
  return()
endif()

find_package(hip QUIET
  HINTS "${ROCM_PATH}" ENV ROCM_PATH ENV HIP_PATH PATHS /opt/rocm
  PATH_SUFFIXES lib/cmake/hip lib64/cmake/hip
  NO_DEFAULT_PATH)
if(NOT hip_FOUND)
  find_package(hip QUIET)
endif()
if(NOT hip_FOUND)
  message(STATUS "HIP: find_package(hip) failed, HIP support disabled.")
  return()
endif()

set(HAVE_HIP 1)
# HIP always requires the modern CMake CUDA language infrastructure.
# With it OFF, OpenCV falls back to the legacy ocv_cuda_compile/FindCUDA
# path which does not understand .hip files and breaks the build.
set(ENABLE_CUDA_FIRST_CLASS_LANGUAGE ON)

if(NOT HAVE_CUDA)
  # No real NVIDIA CUDA installation — HIP operates in standalone mode.
  # Piggyback on the CUDA infrastructure so CUDA-aware OpenCV modules
  # (stitching, etc.) configure and link without a real NVIDIA toolkit.
  set(HAVE_CUDA 1)
  set(HAVE_HIP_STANDALONE 1)
  # Stub CUDA CMake imported targets so modules that link CUDA::cudart
  # (e.g. stitching) satisfy cmake's check without a real NVIDIA toolkit.
  foreach(_cuda_stub cudart cudart_static)
    if(NOT TARGET CUDA::${_cuda_stub})
      add_library(CUDA::${_cuda_stub} INTERFACE IMPORTED GLOBAL)
    endif()
  endforeach()
  unset(_cuda_stub)
endif()

# Map HIP include directories to CUDAToolkit_INCLUDE_DIRS so CUDA-aware
# contrib modules (e.g. hfs) that call ocv_module_include_directories() with
# ${CUDAToolkit_INCLUDE_DIRS} before ocv_define_module() still receive a
# non-empty argument list and avoid a zero-argument call to
# ocv_target_include_modules().
get_target_property(_hip_iface_dirs hip::device INTERFACE_INCLUDE_DIRECTORIES)
if(_hip_iface_dirs)
  set(CUDAToolkit_INCLUDE_DIRS ${_hip_iface_dirs})
else()
  set(CUDAToolkit_INCLUDE_DIRS "${hip_INCLUDE_DIRS}")
endif()
unset(_hip_iface_dirs)

set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
  set(_hip_default_archs "gfx90a;gfx1100;gfx1101;gfx1201;gfx1036")
  set(_hip_detected_archs "")
  find_program(OPENCV_AMDGPU_ARCH_EXECUTABLE amdgpu-arch
    HINTS "${ROCM_PATH}" ENV ROCM_PATH PATHS /opt/rocm
    PATH_SUFFIXES bin llvm/bin)
  if(NOT OPENCV_AMDGPU_ARCH_EXECUTABLE)
    find_program(OPENCV_ROCM_AGENT_ENUMERATOR rocm_agent_enumerator
      HINTS "${ROCM_PATH}" ENV ROCM_PATH PATHS /opt/rocm PATH_SUFFIXES bin)
  endif()
  if(OPENCV_AMDGPU_ARCH_EXECUTABLE)
    execute_process(COMMAND "${OPENCV_AMDGPU_ARCH_EXECUTABLE}"
      OUTPUT_VARIABLE _amdgpu_arch_out OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _amdgpu_arch_res ERROR_QUIET)
    if(_amdgpu_arch_res EQUAL 0 AND _amdgpu_arch_out)
      string(REPLACE "\n" ";" _amdgpu_arch_list "${_amdgpu_arch_out}")
      list(REMOVE_DUPLICATES _amdgpu_arch_list)
      foreach(_a IN LISTS _amdgpu_arch_list)
        if(_a MATCHES "^gfx[0-9a-fA-F]+$")
          list(APPEND _hip_detected_archs "${_a}")
        endif()
      endforeach()
    endif()
  elseif(OPENCV_ROCM_AGENT_ENUMERATOR)
    execute_process(COMMAND "${OPENCV_ROCM_AGENT_ENUMERATOR}"
      OUTPUT_VARIABLE _agent_out OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _agent_res ERROR_QUIET)
    if(_agent_res EQUAL 0 AND _agent_out)
      string(REPLACE "\n" ";" _agent_list "${_agent_out}")
      foreach(_a IN LISTS _agent_list)
        if(_a MATCHES "^gfx[0-9a-fA-F]+$" AND NOT _a STREQUAL "gfx000")
          list(APPEND _hip_detected_archs "${_a}")
        endif()
      endforeach()
      if(_hip_detected_archs)
        list(REMOVE_DUPLICATES _hip_detected_archs)
      endif()
    endif()
  endif()
  if(_hip_detected_archs)
    set(CMAKE_HIP_ARCHITECTURES "${_hip_detected_archs}" CACHE STRING "HIP architectures to build for" FORCE)
  else()
    set(CMAKE_HIP_ARCHITECTURES "${_hip_default_archs}" CACHE STRING "HIP architectures to build for" FORCE)
  endif()
endif()

add_definitions(-D__HIP_PLATFORM_AMD__)
# CMake's HIP language does not propagate INTERFACE_SYSTEM_INCLUDE_DIRECTORIES from
# linked targets (e.g. hip::amdhip64) into HIP_INCLUDES. Add ROCm headers explicitly.
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -I${hip_INCLUDE_DIRS}")

# find_package(hipblas QUIET)
# if(hipblas_FOUND)
#   set(HAVE_CUBLAS 1)
# endif()
# find_package(hipfft QUIET)
# if(hipfft_FOUND)
#   set(HAVE_CUFFT 1)
# endif()

# if(WITH_ROCDECODE)
#   find_package(rocdecode QUIET)
#   if(rocdecode_FOUND)
#     set(HAVE_ROCDECODE 1)
#     set(ROCDECODE_LIBRARIES "${rocdecode_LIBRARIES}")
#   endif()
# endif()

set(CUDA_VERSION_STRING "${hip_VERSION}")
if(NOT CUDA_VERSION)
  set(CUDA_VERSION "${hip_VERSION}")
endif()

# The NVIDIA CUDA-specific contrib modules depend on NVIDIA-proprietary
# libraries (NPP, cuBLAS, cuDNN, etc.) that have no HIP/ROCm equivalents.
# Disable them by default so they do not cause configure errors; users who
# have ported these modules independently can re-enable them explicitly.
foreach(_nv_mod
    cudaarithm cudabgsegm cudacodec cudafeatures2d cudafilters
    cudaimgproc cudalegacy cudaobjdetect cudaoptflow cudastereo cudawarping)
  if(NOT DEFINED BUILD_opencv_${_nv_mod} OR BUILD_opencv_${_nv_mod})
    set(BUILD_opencv_${_nv_mod} OFF CACHE BOOL
        "Disabled for HIP builds (requires NVIDIA CUDA libraries)" FORCE)
  endif()
endforeach()
unset(_nv_mod)

include("${OpenCV_SOURCE_DIR}/cmake/OpenCVDetectCUDAUtils.cmake")

message(STATUS "HIP: ROCm ${hip_VERSION}, building for ${CMAKE_HIP_ARCHITECTURES}")
