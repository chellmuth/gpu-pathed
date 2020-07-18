cmake_minimum_required(VERSION 3.10)
project(rays LANGUAGES CUDA CXX C)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB SOURCES "src/*.cu")
include_directories(include vendor/include)

# Skip LTO to avoid issue with nvcc passthrough
set(PYBIND11_LTO_CXX_FLAGS "")

add_library(rays STATIC ${SOURCES})
set_property(TARGET rays PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET rays PROPERTY CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(pybind11)
pybind11_add_module(path_tracer bindings/path_tracer_bindings.cu)
target_include_directories(path_tracer PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(path_tracer rays)

find_package(Qt5 COMPONENTS Widgets REQUIRED)

file(GLOB APP_SRCS "app/*.cpp")
add_executable(app ${APP_SRCS})
target_include_directories(app PRIVATE app ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(app rays Qt5::Core Qt5::Widgets)
