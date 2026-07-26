cmake_minimum_required(VERSION 3.10)
project(SequentialImagePipeline)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build by default for meaningful timing numbers.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main_sequential
    main_sequential.cpp
    grayscale.cpp
    histogram.cpp
)

target_link_libraries(main_sequential ${OpenCV_LIBS})
