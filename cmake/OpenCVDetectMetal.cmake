# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

# Metal backend of the non-CPU HAL. Apple platforms only, and only where the toolchain can
# compile Objective-C++: the backend is a .mm file, which the module source glob already picks
# up (OpenCVModule.cmake), so all this has to establish is that it will build and link.

if(NOT APPLE)
  message(STATUS "Metal: not an Apple platform, Metal support disabled.")
  return()
endif()

# CMake 3.16+ knows OBJCXX as a language. Without it the .mm file would be handed to the C++
# compiler, which does not understand Objective-C syntax.
if(NOT DEFINED CMAKE_OBJCXX_COMPILER)
  message(STATUS "Metal: OBJCXX is not enabled as a language, Metal support disabled.")
  return()
endif()

find_library(APPLE_METAL_FRAMEWORK Metal)
find_library(APPLE_FOUNDATION_FRAMEWORK Foundation)

if(NOT APPLE_METAL_FRAMEWORK OR NOT APPLE_FOUNDATION_FRAMEWORK)
  message(STATUS "Metal: the Metal or Foundation framework was not found, Metal support disabled.")
  return()
endif()

# MTLSharedEvent, which the backend uses for Event, arrived in macOS 10.14 / iOS 12. Older
# deployment targets would compile and then fail at runtime on the first record().
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_OSX_DEPLOYMENT_TARGET
   AND CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS "10.14")
  message(STATUS "Metal: the deployment target ${CMAKE_OSX_DEPLOYMENT_TARGET} predates MTLSharedEvent "
                 "(10.14), Metal support disabled.")
  return()
endif()

set(HAVE_METAL 1)
set(METAL_LIBRARIES "${APPLE_METAL_FRAMEWORK}" "${APPLE_FOUNDATION_FRAMEWORK}")

message(STATUS "Metal: building the accel backend (${APPLE_METAL_FRAMEWORK})")
