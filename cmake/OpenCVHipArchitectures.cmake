# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

function(ocv_hip_match_archs _text _regex _out)
  string(REPLACE "\n" ";" _list "${_text}")
  string(REPLACE "," ";" _list "${_list}")
  set(_archs "")
  foreach(_a IN LISTS _list)
    string(STRIP "${_a}" _a)
    if(_a MATCHES "${_regex}" AND NOT _a STREQUAL "gfx000")
      list(APPEND _archs "${_a}")
    endif()
  endforeach()
  if(_archs)
    list(REMOVE_DUPLICATES _archs)
  endif()
  set(${_out} "${_archs}" PARENT_SCOPE)
endfunction()

# Build for the GPUs present, or for whatever family-wide targets this toolchain
# reports. Both adapt across ROCm versions, so nothing here is hardcoded.
macro(ocv_hip_select_architectures)
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
    set(_hip_archs "")

    find_program(OPENCV_AMDGPU_ARCH_EXECUTABLE NAMES amdgpu-arch offload-arch
      HINTS "${ROCM_PATH}" ENV ROCM_PATH PATHS /opt/rocm
      PATH_SUFFIXES bin llvm/bin)
    if(OPENCV_AMDGPU_ARCH_EXECUTABLE)
      execute_process(COMMAND "${OPENCV_AMDGPU_ARCH_EXECUTABLE}"
        OUTPUT_VARIABLE _out OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _res ERROR_QUIET)
      if(_res EQUAL 0)
        ocv_hip_match_archs("${_out}" "^gfx[0-9a-f]+$" _hip_archs)
      endif()
    endif()

    if(NOT _hip_archs)
      find_program(OPENCV_ROCM_AGENT_ENUMERATOR rocm_agent_enumerator
        HINTS "${ROCM_PATH}" ENV ROCM_PATH PATHS /opt/rocm PATH_SUFFIXES bin)
      if(OPENCV_ROCM_AGENT_ENUMERATOR)
        execute_process(COMMAND "${OPENCV_ROCM_AGENT_ENUMERATOR}"
          OUTPUT_VARIABLE _out OUTPUT_STRIP_TRAILING_WHITESPACE
          RESULT_VARIABLE _res ERROR_QUIET)
        if(_res EQUAL 0)
          ocv_hip_match_archs("${_out}" "^gfx[0-9a-f]+$" _hip_archs)
        endif()
      endif()
    endif()

    # No GPU to look at (CI, packaging, cross builds): ask the compiler which
    # generic targets it can emit. Each covers a whole family.
    if(NOT _hip_archs)
      execute_process(COMMAND "${CMAKE_HIP_COMPILER}"
        --target=amdgcn-amd-amdhsa -print-supported-cpus
        OUTPUT_VARIABLE _out ERROR_VARIABLE _out
        RESULT_VARIABLE _res OUTPUT_STRIP_TRAILING_WHITESPACE)
      if(_res EQUAL 0)
        ocv_hip_match_archs("${_out}" "^gfx[0-9]+(-[0-9]+)?-generic$" _hip_archs)
      endif()
      if(_hip_archs)
        message(STATUS "HIP: no GPU detected, building for generic targets.")
      endif()
    endif()

    if(NOT _hip_archs)
      message(FATAL_ERROR "HIP: no GPU found and the compiler reported no generic targets. "
        "Check that ROCm is installed and amdgpu-arch works.")
    endif()

    set(CMAKE_HIP_ARCHITECTURES "${_hip_archs}" CACHE STRING "HIP architectures to build for" FORCE)
    unset(_hip_archs)
    unset(_out)
    unset(_res)
  endif()
endmacro()
