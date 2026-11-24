cmake_minimum_required(VERSION 3.18)
project(EdgeVision LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 4060 (Ada Lovelace) mimarisi: sm_89
set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    /usr/include/x86_64-linux-gnu
    include
)

add_executable(edge_vision
    src/main.cpp
    src/yolo_trt.cpp
    src/bytetrack.cpp
    src/preprocess.cu
)

target_link_libraries(edge_vision
    ${OpenCV_LIBS}
    CUDA::cudart
    nvinfer
)