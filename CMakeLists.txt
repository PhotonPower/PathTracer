cmake_minimum_required(VERSION 3.25)

project(PathTracer
  VERSION 0.1.0
  DESCRIPTION "Light transport on trimmed NURBS geometry: light tracing, path tracing, bidirectional, photon mapping"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(PT_BUILD_PYTHON "Build the Python package pathtracer (module pathtracer._core)" ON)
option(PT_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

find_package(Threads REQUIRED)

add_library(pathtracer STATIC
  src/core.cpp src/mesh.cpp src/bvh.cpp src/material.cpp src/photometry.cpp
  src/scene.cpp src/sensor.cpp src/integrator.cpp src/photon.cpp)
add_library(PathTracer::pathtracer ALIAS pathtracer)
target_include_directories(pathtracer PUBLIC include PRIVATE src)
target_link_libraries(pathtracer PUBLIC Threads::Threads)
set_target_properties(pathtracer PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
  add_compile_options(/utf-8)  # the sources are UTF-8 (messages with °, …)
  target_compile_options(pathtracer PRIVATE /W4 $<$<BOOL:${PT_WARNINGS_AS_ERRORS}>:/WX>)
  target_compile_definitions(pathtracer PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(pathtracer PRIVATE -Wall -Wextra $<$<BOOL:${PT_WARNINGS_AS_ERRORS}>:-Werror>)
endif()

if(PT_BUILD_PYTHON)
  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core bindings/module.cpp)
  target_link_libraries(_core PRIVATE pathtracer)
  # next to the Python sources, so that the package runs from the source tree after a build
  set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/pathtracer"
                                          LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/python/pathtracer")
  install(TARGETS _core LIBRARY DESTINATION pathtracer)
endif()
