cmake_minimum_required(VERSION 3.10)
project(clip_retrieval)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    /usr/include/aarch64-linux-gnu
)

cuda_add_executable(clip_retrieval clip_retrieval.cu)

target_link_libraries(clip_retrieval
    ${OpenCV_LIBS}
    nvinfer
    ${CUDA_LIBRARIES}
)
