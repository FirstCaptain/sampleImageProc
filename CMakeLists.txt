cmake_minimum_required(VERSION 3.22)

# set the project name
project(sample LANGUAGES CXX C)

# export symbols so backtraces are useful
set(CMAKE_ENABLE_EXPORTS ON)

set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${OpenCV_LIBRARY_DIRS})

# Turn on VL_DEBUG for Debug builds
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DVL_DEBUG")

set(TARGET_NAME "sample")

set(SOURCES
  src/main.cpp
)


add_executable(${TARGET_NAME} ${SOURCES})

target_include_directories(${TARGET_NAME}
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/
)

target_link_libraries(${TARGET_NAME}
  PUBLIC
  ${OpenCV_LIBS}
)