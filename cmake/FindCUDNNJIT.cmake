# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.


if(CUDA_FOUND)
  find_cuda_helper_libs(cudnn)
  find_cuda_helper_libs(cudnn_graph)
  find_cuda_helper_libs(cudnn_engines_runtime_compiled)
  set(CUDNN_SHIM_LIBRARY        ${CUDA_cudnn_LIBRARY}                           CACHE FILEPATH "location of the cuDNN dispatch shim library")
  set(CUDNN_GRAPH_LIBRARY       ${CUDA_cudnn_graph_LIBRARY}                     CACHE FILEPATH "location of the cuDNN graph library")
  set(CUDNN_ENGINES_RTC_LIBRARY ${CUDA_cudnn_engines_runtime_compiled_LIBRARY}  CACHE FILEPATH "location of the cuDNN runtime-compiled engines library")
  unset(CUDA_cudnn_LIBRARY CACHE)
  unset(CUDA_cudnn_graph_LIBRARY CACHE)
  unset(CUDA_cudnn_engines_runtime_compiled_LIBRARY CACHE)
endif()

if(CUDNN_GRAPH_LIBRARY)
  find_path(CUDNNJIT_INCLUDE_DIR
    cudnn_graph.h
    PATHS ${CUDA_TOOLKIT_INCLUDE} ${CUDNN_INCLUDE_DIR}
    DOC "location of cudnn_graph.h"
    NO_DEFAULT_PATH
  )

  if(NOT CUDNNJIT_INCLUDE_DIR)
    find_path(CUDNNJIT_INCLUDE_DIR
      cudnn_graph.h
      DOC "location of cudnn_graph.h"
    )
  endif()
endif()

if(CUDNNJIT_INCLUDE_DIR AND EXISTS "${CUDNNJIT_INCLUDE_DIR}/cudnn_version.h")
  file(READ "${CUDNNJIT_INCLUDE_DIR}/cudnn_version.h" CUDNN_H_CONTENTS)
  string(REGEX MATCH "define CUDNN_MAJOR ([0-9]+)" _ "${CUDNN_H_CONTENTS}")
  set(CUDNN_VERSION_MAJOR ${CMAKE_MATCH_1} CACHE INTERNAL "")
  string(REGEX MATCH "define CUDNN_MINOR ([0-9]+)" _ "${CUDNN_H_CONTENTS}")
  set(CUDNN_VERSION_MINOR ${CMAKE_MATCH_1} CACHE INTERNAL "")
  string(REGEX MATCH "define CUDNN_PATCHLEVEL ([0-9]+)" _ "${CUDNN_H_CONTENTS}")
  set(CUDNN_VERSION_PATCH ${CMAKE_MATCH_1} CACHE INTERNAL "")
  set(CUDNN_VERSION "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
  unset(CUDNN_H_CONTENTS)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDNNJIT
  FOUND_VAR CUDNNJIT_FOUND
  REQUIRED_VARS
    CUDNN_SHIM_LIBRARY
    CUDNN_GRAPH_LIBRARY
    CUDNN_ENGINES_RTC_LIBRARY
    CUDNNJIT_INCLUDE_DIR
  VERSION_VAR CUDNN_VERSION
)

if(CUDNNJIT_FOUND)
  set(CUDNNJIT_LIBRARIES    ${CUDNN_SHIM_LIBRARY} ${CUDNN_GRAPH_LIBRARY} ${CUDNN_ENGINES_RTC_LIBRARY})
  set(CUDNNJIT_INCLUDE_DIRS ${CUDNNJIT_INCLUDE_DIR})
endif()

mark_as_advanced(
  CUDNN_SHIM_LIBRARY
  CUDNN_GRAPH_LIBRARY
  CUDNN_ENGINES_RTC_LIBRARY
  CUDNNJIT_INCLUDE_DIR
)
