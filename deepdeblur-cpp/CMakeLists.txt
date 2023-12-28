cmake_minimum_required(VERSION 3.10)
project(deepdeblur)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")

set(OpenCV_DIR "/opt/thundersoft/opencv-4.5.5/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

include_directories(
    ./
    ${OpenCV_INCLUDE_DIRS}
    /usr/include/SNPE
)

add_executable(
    main
    ./main.cpp
    ./SNPETask.cpp
)

link_directories(
    main
    ${OpenCV_LIBRARIES}
    /usr/lib/libSNPE.so
)

target_link_libraries(
    main
    pthread
    dl
    ${OpenCV_LIBRARIES}
    /usr/lib/libSNPE.so
)
