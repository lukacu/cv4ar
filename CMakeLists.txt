PROJECT(cv4ar)
CMAKE_MINIMUM_REQUIRED(VERSION 3.3 FATAL_ERROR)

FIND_PACKAGE(OpenCV REQUIRED)
LINK_LIBRARIES(${OpenCV_LIBS})
INCLUDE_DIRECTORIES(AFTER ${OpenCV_INCLUDE_DIR})

IF ("opencv_xfeatures2d" IN_LIST OpenCV_LIBS)
	ADD_DEFINITIONS(-D_XFEATURES)
ENDIF()

IF(MSVC)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
ELSEIF(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
ENDIF()

ADD_EXECUTABLE(threshold src/threshold.cpp)
ADD_EXECUTABLE(filtering src/filtering.cpp)
ADD_EXECUTABLE(edges src/edges.cpp)
ADD_EXECUTABLE(contours src/contours.cpp)
ADD_EXECUTABLE(detection src/detection.cpp src/marker.cpp)
ADD_EXECUTABLE(homography src/homography.cpp src/marker.cpp)
ADD_EXECUTABLE(keypoints src/keypoints.cpp)
ADD_EXECUTABLE(correspondences src/correspondences.cpp)
ADD_EXECUTABLE(ransac src/ransac.cpp)
ADD_EXECUTABLE(pose src/pose.cpp src/marker.cpp)

FILE(COPY
    data/webcam.xml
    data/video.mp4
    data/marker1.png
    data/marker2.png
    data/marker3.png
    data/marker4.png
    data/marker5.png
    data/marker6.png
DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

