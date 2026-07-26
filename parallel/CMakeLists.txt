cmake_minimum_required(VERSION 3.10)
project(ImagePipelinePthreads CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# pthreads ships with macOS/Linux by default -- no brew install needed.
find_package(Threads REQUIRED)

add_executable(main_parallel
    main_parallel.cpp
    grayscale_parallel.cpp
    histogram_parallel.cpp
    equalize_parallel.cpp
)

target_link_libraries(main_parallel ${OpenCV_LIBS} Threads::Threads)
