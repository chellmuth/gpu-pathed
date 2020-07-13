cmake_minimum_required(VERSION 3.10)
project(rays LANGUAGES CUDA CXX C)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB SOURCES "src/*.cu")
include_directories(include vendor/include)

# Skip LTO to avoid issue with nvcc passthrough
set(PYBIND11_LTO_CXX_FLAGS "")

add_subdirectory(pybind11)
pybind11_add_module(path_tracer bindings/path_tracer_bindings.cu ${SOURCES})
# set_target_properties(path_tracer PROPERTIES LINKER_LANGUAGE CXX)
set_property(TARGET path_tracer PROPERTY CUDA_SEPARABLE_COMPILATION ON)
