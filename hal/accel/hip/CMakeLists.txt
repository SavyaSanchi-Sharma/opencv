# HIP backend of the non-CPU HAL (AMD GPUs), as a plugin: libopencv_core_accel_hip.so
#
#   cmake -S hal/accel/hip -B build_accel_hip \
#         -DOpenCV_DIR=<OpenCV build directory> \
#         -DCMAKE_PREFIX_PATH=/opt/rocm
#   cmake --build build_accel_hip
#
# CMAKE_HIP_ARCHITECTURES is detected from the GPUs present, or from the toolchain's
# generic targets when there is none; set it explicitly to override.
#
# Put the library next to the OpenCV libraries, in a directory of OPENCV_CORE_PLUGIN_PATH, or
# load it with cv::accel::loadPlugin(). It only loads into the OpenCV version it was built against.

cmake_minimum_required(VERSION 3.21)
project(opencv_core_accel_hip CXX HIP)

get_filename_component(OpenCV_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
include("${OpenCV_SOURCE_DIR}/cmake/OpenCVPluginStandalone.cmake")

find_package(hip REQUIRED)

include("${OpenCV_SOURCE_DIR}/cmake/OpenCVHipArchitectures.cmake")
ocv_hip_select_architectures()

include("${OpenCV_SOURCE_DIR}/cmake/OpenCVDetectROCm.cmake")

set(OPENCV_PLUGIN_EXTRA_SRC_FILES
    "${CMAKE_CURRENT_LIST_DIR}/core/core.hip"
    "${CMAKE_CURRENT_LIST_DIR}/core/flip.hip"
    "${CMAKE_CURRENT_LIST_DIR}/core/transpose.hip")
set(OPENCV_PLUGIN_EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}")

ocv_create_plugin(core "opencv_core_accel_hip" "hip::host" "HIP"
                  "src/accel/backend_hip.cpp")

target_compile_definitions(${OPENCV_PLUGIN_NAME} PRIVATE HAVE_ACCEL_HIP=1)
if(ROCM_FOUND)
  target_compile_definitions(${OPENCV_PLUGIN_NAME} PRIVATE
      OPENCV_ROCM_VERSION_MAJOR=${ROCM_VERSION_MAJOR}
      OPENCV_ROCM_VERSION_MINOR=${ROCM_VERSION_MINOR})
  message(STATUS "accel_hip: ROCm ${ROCM_VERSION}, HIP ${hip_VERSION}, building for ${CMAKE_HIP_ARCHITECTURES}")
endif()
