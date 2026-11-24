cmake_minimum_required(VERSION 3.18)
project(edge_vision_ros LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(EDGE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

include_directories(
  ${EDGE_ROOT}/include
  ${OpenCV_INCLUDE_DIRS}
  /usr/include/x86_64-linux-gnu
)

add_executable(edge_vision_node
  src/edge_vision_node.cpp
  ${EDGE_ROOT}/src/yolo_trt.cpp
  ${EDGE_ROOT}/src/bytetrack.cpp
  ${EDGE_ROOT}/src/preprocess.cu
)

ament_target_dependencies(edge_vision_node
  rclcpp
  sensor_msgs
  vision_msgs
  cv_bridge
)

target_link_libraries(edge_vision_node
  ${OpenCV_LIBS}
  CUDA::cudart
  nvinfer
)

install(TARGETS edge_vision_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()