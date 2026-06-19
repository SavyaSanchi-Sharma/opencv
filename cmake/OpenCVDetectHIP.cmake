# Detect and configure AMD ROCm/HIP support.
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.

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

# State matrix for downstream code (this file, OpenCVDetectCUDAUtils.cmake,
# module CMakeLists.txts) to distinguish build configurations:
#
#   State              HAVE_HIP   HAVE_CUDA      HAVE_HIP_STANDALONE
#   ─────────────────  ────────   ────────────   ───────────────────
#   HIP only           1          1 (faked)       1
#   HIP + real CUDA    1          1 (real)        unset
#   CUDA only          unset      1 (real)        unset
#
if(NOT HAVE_CUDA AND NOT CMAKE_CUDA_COMPILER)
  # No real NVIDIA CUDA installation — HIP operates in standalone mode.
  # Piggyback on the CUDA infrastructure so CUDA-aware OpenCV modules
  # (stitching, etc.) configure and link without a real NVIDIA toolkit.
  # NOTE: check CMAKE_CUDA_COMPILER (set by enable_language(CUDA) at CMakeLists.txt
  # line ~647) rather than HAVE_CUDA alone — HAVE_CUDA is set later at line ~783
  # by OpenCVFindLibsPerf → OpenCVDetectCUDALanguage.  Using only NOT HAVE_CUDA
  # would always trigger standalone mode even when nvcc is present.
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
# In combined mode find_package(CUDAToolkit) already set the real CUDA path;
# overwriting it with /opt/rocm/include would break CUDA compilation.
if(HAVE_HIP_STANDALONE)
  get_target_property(_hip_iface_dirs hip::device INTERFACE_INCLUDE_DIRECTORIES)
  if(_hip_iface_dirs)
    set(CUDAToolkit_INCLUDE_DIRS ${_hip_iface_dirs})
  else()
    set(CUDAToolkit_INCLUDE_DIRS "${hip_INCLUDE_DIRS}")
  endif()
  unset(_hip_iface_dirs)
endif()

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

# In standalone mode all source files are compiled by either g++ or amdclang++,
# so a blanket -D is fine.  In combined mode the .cu files are compiled by nvcc
# and must NOT see __HIP_PLATFORM_AMD__ (it would pull in ROCm-only intrinsics).
add_compile_options($<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-D__HIP_PLATFORM_AMD__>)
if(NOT HAVE_HIP_STANDALONE)
  # hip::device's INTERFACE_COMPILE_DEFINITIONS propagates -D__HIP_PLATFORM_AMD__=1
  # into nvcc's CUDA_DEFINES (which appear before CUDA_FLAGS on the command line).
  # Undo it and define the correct NVIDIA platform macro via CUDA_FLAGS, which is
  # appended after CUDA_DEFINES.
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -U__HIP_PLATFORM_AMD__ -D__HIP_PLATFORM_NVIDIA__")
  # cmake suppresses explicit -I entries for paths already in
  # CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES (treats them as redundant). But those
  # implicit paths are searched by nvcc AFTER the explicit -isystem entries, so
  # ROCm's thrust at /opt/rocm/include/thrust/ wins over NVIDIA's cccl/thrust/.
  # Fix: find the CUDA cccl dir (where NVIDIA's thrust lives), remove it from the
  # implicit list so cmake emits it as an explicit -I.  All -I entries in the
  # CUDA RSP precede -isystem entries, so nvcc finds NVIDIA thrust first.
  foreach(_cuda_inc ${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES})
    if(EXISTS "${_cuda_inc}/thrust")
      set(OPENCV_HIP_CUDA_THRUST_INCLUDE "${_cuda_inc}" CACHE INTERNAL "" FORCE)
      list(REMOVE_ITEM CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "${_cuda_inc}")
      break()
    endif()
  endforeach()
  unset(_cuda_inc)
endif()
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

# In standalone mode there is no NVIDIA CUDA toolkit, so the NVIDIA-specific
# contrib modules (which link NPP, cuBLAS, cuDNN, etc.) cannot build.
# In combined mode (HIP + real CUDA) those modules are available normally.
if(HAVE_HIP_STANDALONE)
  foreach(_nv_mod
      cudaarithm cudabgsegm cudacodec cudafeatures2d cudafilters
      cudaimgproc cudalegacy cudaobjdetect cudaoptflow cudastereo cudawarping)
    if(NOT DEFINED BUILD_opencv_${_nv_mod} OR BUILD_opencv_${_nv_mod})
      set(BUILD_opencv_${_nv_mod} OFF CACHE BOOL
          "Disabled: HIP standalone build has no NVIDIA CUDA libraries" FORCE)
    endif()
  endforeach()
  unset(_nv_mod)
endif()

include("${OpenCV_SOURCE_DIR}/cmake/OpenCVDetectCUDAUtils.cmake")

message(STATUS "HIP: ROCm ${hip_VERSION}, building for ${CMAKE_HIP_ARCHITECTURES}")
