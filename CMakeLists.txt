cmake_minimum_required(VERSION 2.8.3)
project(disparity_camera)

add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  message_filters
  image_transport
  roscpp
  sensor_msgs
  std_msgs
)

find_package(OpenCV 3.4 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS  ${CATKIN_DEPENDENCIES}
  DEPENDS OpenCV
)

include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME}_sgbm src/disparity_sgbm.cpp)
target_link_libraries(${PROJECT_NAME}_sgbm ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

