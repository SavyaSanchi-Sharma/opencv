# opencv_hip_tune times the HIP plugin's kernels, so it exists only where the plugin is built.
if(NOT HAVE_HIP OR NOT BUILD_PLUGIN_ACCEL_HIP OR NOT BUILD_SHARED_LIBS OR NOT UNIX)
  return()
endif()

ocv_add_application(opencv_hip_tune
    MODULES opencv_core opencv_imgproc
    SRCS opencv_hip_tune.cpp)

if(TARGET opencv_hip_tune)
  # hipGetDeviceProperties / hipRuntimeGetVersion for the stamp of the tuned file; no device code
  ocv_target_link_libraries(opencv_hip_tune hip::host ${CMAKE_DL_LIBS})
endif()
