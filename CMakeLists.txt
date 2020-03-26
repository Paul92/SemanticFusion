cmake_minimum_required(VERSION 3.12.2)
if(POLICY CMP0007)
  cmake_policy(SET CMP0007 NEW)
endif()

if (NOT CUDA_HOST_COMPILER)
    message(SEND_ERROR "CUDA_HOST_COMPILER not set")
endif()
if (NOT CUDA_ARCH_BIN)
    message(SEND_ERROR "CUDA_ARCH_BIN not set")
endif()

set(CUDA_PROPAGATE_HOST_FLAGS "NO")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(CUDA REQUIRED)
find_package(Glog REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenNI2 REQUIRED)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/elasticfusionpublic/Core/src")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/caffe_semanticfusion")


file(GLOB_RECURSE srcs src/*.cpp)
file(GLOB_RECURSE cuda src/*.cu)

list(REMOVE_ITEM srcs "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

include(CudaComputeTargetFlags.cmake)
APPEND_TARGET_ARCH_FLAGS()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-std=c++11 --ftz=true;--prec-div=false;--prec-sqrt=false") 
set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -g -msse2 -msse3 -Wall -std=c++11")

cuda_add_library(semantic_fusion_cuda_library SHARED ${cuda})

add_library(semanticfusion-original SHARED
           "${CMAKE_CURRENT_SOURCE_DIR}/semanticfusion-library.cpp"
           ${srcs}
)

target_link_libraries(semanticfusion-original
                      ${CUDA_LIBRARIES}
                      ${GLOG_LIBRARY}
                      ${OpenCV_LIBS}
                      ${OPENNI2_LIBRARY}
                      semantic_fusion_cuda_library
                      efusion
                      caffe
)

target_include_directories(semanticfusion-original PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
                           ${EIGEN_INCLUDE_DIRS}
                           ${CUDA_INCLUDE_DIRS}
                           ${GLOG_INCLUDE_DIR}
                           ${OPENNI2_INCLUDE_DIR}
)

generate_slam_library(semanticfusion-original "")

