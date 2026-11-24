cmake_minimum_required(VERSION 3.18)
project(EdgeVision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV ve CUDA kütüphanelerini bul
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

# TensorRT başlık dosyalarının ve kütüphanelerinin yolu
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    /usr/include/x86_64-linux-gnu
    include
)

# Çalıştırılabilir dosyamızı tanımla
add_executable(edge_vision src/main.cpp src/yolo_trt.cpp src/bytetrack.cpp)

# Kütüphaneleri projeye bağla
target_link_libraries(edge_vision
    ${OpenCV_LIBS}
    CUDA::cudart
    nvinfer
)