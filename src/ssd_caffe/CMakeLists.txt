cmake_minimum_required(VERSION 2.8.3)
project(ssd_caffe)

	
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cmake_modules roscpp roslib std_msgs sensor_msgs image_transport cv_bridge message_generation
)

set(CAFFE_INCLUDEDIR caffe/include /usr/local/cuda/include)
set(CAFFE_LINK_LIBRARY caffe/lib /usr/local/cuda/lib64)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenCV REQUIRED)

 generate_messages(
   DEPENDENCIES
   std_msgs
 )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES gflags
#  CATKIN_DEPENDS roscpp
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
include_directories(${CAFFE_INCLUDEDIR})

## Declare a cpp executable
## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
## Specify libraries to link a library or executable target against


link_directories(${CAFFE_LINK_LIBRARY})

add_executable(ssd_all_bbox src/ssd_all_bbox.cpp)
add_dependencies(ssd_all_bbox ssd_caffe_generate_messages_cpp)
target_link_libraries(ssd_all_bbox 
   ${catkin_LIBRARIES}
   ${OpenCV_LIBRARIES}
   glog gflags caffe		
 )


