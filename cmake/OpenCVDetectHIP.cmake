# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

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
# HIP needs the first-class CUDA language; legacy FindCUDA can't compile .hip files.
set(ENABLE_CUDA_FIRST_CLASS_LANGUAGE ON)

set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# CMAKE_HIP_ARCHITECTURES is chosen before enable_language(HIP), in the root CMakeLists.

# Define __HIP_PLATFORM_AMD__ for non-CUDA compiles only (nvcc must not see it).
add_compile_options($<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-D__HIP_PLATFORM_AMD__>)
if(HAVE_CUDA)
  # hip::device forces -D__HIP_PLATFORM_AMD__=1 onto nvcc; undo it and set the
  # NVIDIA platform macro via CUDA_FLAGS (appended after CUDA_DEFINES).
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -U__HIP_PLATFORM_AMD__ -D__HIP_PLATFORM_NVIDIA__")
  # Force NVIDIA's thrust ahead of ROCm's: drop the cccl dir from the implicit
  # include list so cmake emits it as an explicit -I (which precedes -isystem).
  foreach(_cuda_inc ${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES})
    if(EXISTS "${_cuda_inc}/thrust")
      set(OPENCV_HIP_CUDA_THRUST_INCLUDE "${_cuda_inc}" CACHE INTERNAL "" FORCE)
      list(REMOVE_ITEM CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "${_cuda_inc}")
      break()
    endif()
  endforeach()
  unset(_cuda_inc)
endif()
# CMake's HIP language doesn't propagate hip::amdhip64 include dirs; add explicitly.
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -I${hip_INCLUDE_DIRS}")

include("${OpenCV_SOURCE_DIR}/cmake/OpenCVDetectROCm.cmake")

if(ROCM_FOUND)
  add_compile_definitions(OPENCV_ROCM_VERSION_MAJOR=${ROCM_VERSION_MAJOR})
  add_compile_definitions(OPENCV_ROCM_VERSION_MINOR=${ROCM_VERSION_MINOR})
  message(STATUS "HIP: ROCm ${ROCM_VERSION}, HIP ${hip_VERSION}, building for ${CMAKE_HIP_ARCHITECTURES}")
else()
  message(STATUS "HIP: ${hip_VERSION} (ROCm version unknown), building for ${CMAKE_HIP_ARCHITECTURES}")
endif()
