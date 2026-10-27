# Metal backend of the non-CPU HAL (Apple GPUs), as a plugin: libopencv_core_accel_metal.dylib
#
#   cmake -S modules/core/misc/plugins/accel_metal -B build_accel_metal \
#         -DOpenCV_DIR=<OpenCV build directory>
#   cmake --build build_accel_metal
#
# Put the library next to the OpenCV libraries, in a directory of OPENCV_CORE_PLUGIN_PATH, or
# load it with cv::accel::loadPlugin(). It only loads into the OpenCV version it was built against.
#
# Only needed when core itself was built without Metal: with WITH_METAL=ON the backend is part
# of core already and this plugin would register a second copy of the same devices.

cmake_minimum_required(VERSION 3.21)
project(opencv_core_accel_metal CXX OBJCXX)

if(NOT APPLE)
  message(FATAL_ERROR "The Metal backend builds on Apple platforms only.")
endif()

get_filename_component(OpenCV_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../.." ABSOLUTE)
include("${OpenCV_SOURCE_DIR}/cmake/OpenCVPluginStandalone.cmake")

find_library(APPLE_METAL_FRAMEWORK Metal REQUIRED)
find_library(APPLE_FOUNDATION_FRAMEWORK Foundation REQUIRED)

# ocv_create_plugin wants a target it can check with if(TARGET ...) and link, the way the HIP
# plugin passes hip::host. find_library gives paths, so the two frameworks are wrapped in one.
add_library(opencv_metal_frameworks INTERFACE IMPORTED GLOBAL)
set_target_properties(opencv_metal_frameworks PROPERTIES
  INTERFACE_LINK_LIBRARIES "${APPLE_METAL_FRAMEWORK};${APPLE_FOUNDATION_FRAMEWORK}")

ocv_create_plugin(core "opencv_core_accel_metal" "opencv_metal_frameworks" "Metal"
                  "src/accel/backend_metal.mm")
target_compile_definitions(${OPENCV_PLUGIN_NAME} PRIVATE HAVE_ACCEL_METAL=1)

# ocv_create_plugin calls project(... LANGUAGES CXX) of its own, so say outright that this is
# Objective-C++ rather than relying on the .mm extension being mapped. The backend also retains
# and releases its Metal objects by hand, which ARC would do a second time.
set_source_files_properties("${OpenCV_SOURCE_DIR}/modules/core/src/accel/backend_metal.mm"
                            PROPERTIES LANGUAGE OBJCXX
                                       COMPILE_OPTIONS "-fno-objc-arc")
