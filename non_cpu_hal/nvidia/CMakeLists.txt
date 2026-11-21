# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.

cmake_minimum_required(VERSION ${MIN_VER_CMAKE} FATAL_ERROR)

set(NVIDIA_HAL_INCLUDE_DIR include)
set(NVIDIA_HAL_SOURCE_DIR  src)

file(GLOB nvidia_hal_headers RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
     "${NVIDIA_HAL_INCLUDE_DIR}/opencv2/*.hpp")
file(GLOB nvidia_hal_private RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
     "${NVIDIA_HAL_SOURCE_DIR}/*.hpp" "${NVIDIA_HAL_SOURCE_DIR}/cuda/*.hpp")
file(GLOB nvidia_hal_sources RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
     "${NVIDIA_HAL_SOURCE_DIR}/*.cpp" "${NVIDIA_HAL_SOURCE_DIR}/cuda/*.cu")

set(HAL_LIB_NAME "nvidia_hal")

ocv_add_library(${HAL_LIB_NAME} STATIC
                ${nvidia_hal_headers} ${nvidia_hal_private} ${nvidia_hal_sources})

set_target_properties(${HAL_LIB_NAME} PROPERTIES
                      ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH})

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(${HAL_LIB_NAME} EXPORT OpenCVModules
                     ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()

target_include_directories(${HAL_LIB_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CUDA_INCLUDE_DIRS}
  ${CUDA_TOOLKIT_INCLUDE}
  ${CMAKE_SOURCE_DIR}/modules/core/include
  ${CMAKE_SOURCE_DIR}/modules/imgproc/include)

set(NVIDIA_HAL_FOUND        TRUE            CACHE INTERNAL "")
set(NVIDIA_HAL_VERSION      "0.0.1"         CACHE INTERNAL "")
set(NVIDIA_HAL_LIBRARIES    ${HAL_LIB_NAME} CACHE INTERNAL "")
set(NVIDIA_HAL_HEADERS      "opencv2/cuda_hal.hpp" CACHE INTERNAL "")
set(NVIDIA_HAL_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL "")
