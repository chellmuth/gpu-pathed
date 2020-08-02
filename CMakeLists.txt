cmake_minimum_required(VERSION 3.10)
project(rays LANGUAGES CUDA CXX C)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# set(CMAKE_CXX_FLAGS_DEBUG  "-g")

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Skip LTO to avoid issue with nvcc passthrough
set(PYBIND11_LTO_CXX_FLAGS "")

find_package(Boost 1.56 REQUIRED COMPONENTS filesystem)

find_package(OptiX REQUIRED)

add_library(ptx_files OBJECT optix/optix_kernel.cu)
target_include_directories(ptx_files PUBLIC include)
target_include_directories(ptx_files PUBLIC "${OptiX_INCLUDE}")
target_include_directories(ptx_files PUBLIC "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
set_property(TARGET ptx_files PROPERTY CUDA_PTX_COMPILATION ON)
set_property(TARGET ptx_files PROPERTY POSITION_INDEPENDENT_CODE ON)

add_custom_target(
  ptx_headers
  COMMAND ${CMAKE_COMMAND}
  "-DCPP_FILE=${CMAKE_CURRENT_SOURCE_DIR}/optix/optix_kernel.h"
  "-DPTX_FILE=$<TARGET_OBJECTS:ptx_files>"
  "-DVARIABLE_NAME=ptxSource"
  "-DCUDA_BIN2C_EXECUTABLE=bin2c"
  -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ptx2cpp.cmake"
  DEPENDS ptx_files
  COMMENT "Convert ptx files to header file"
)

add_library(rays STATIC
  src/camera.cu
  src/cuda_globals.cu
  src/framebuffer.cu
  src/hit_test.cu
  src/path_tracer.cu
  src/primitive.cu
  src/render_session.cu
  src/scene.cu
  src/scene_data.cu
  src/scene_model.cu
  src/sphere.cu
  src/spp_optimizer.cpp
  src/transform.cu
  src/triangle.cu

  src/materials/dummy.cu
  src/materials/material.cu
  src/materials/material_table.cpp

  src/renderers/g_buffer.cu
  src/renderers/optix.cpp
  src/renderers/optix_tracer.cu

  src/parsers/mtl_parser.cpp
  src/parsers/obj_parser.cpp
  src/parsers/string_util.cpp
)

target_include_directories(rays PUBLIC include optix)
target_include_directories(rays PUBLIC "${OptiX_INCLUDE}")
target_include_directories(rays PUBLIC "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
target_link_libraries(rays Boost::filesystem)
set_property(TARGET rays PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET rays PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# target_compile_options(rays PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-D__CUDACC__>)
add_dependencies(rays ptx_headers)

add_subdirectory(pybind11)
pybind11_add_module(path_tracer bindings/path_tracer_bindings.cu)
target_link_libraries(path_tracer rays)

find_package(Qt5 COMPONENTS Widgets REQUIRED)

# Test application for profiling
add_executable(app
  app/main.cpp
  app/gl_widget.cpp
)
target_include_directories(app PRIVATE app)
target_link_libraries(app rays Qt5::Core Qt5::Widgets)
