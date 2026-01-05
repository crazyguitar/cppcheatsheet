cmake_minimum_required(VERSION 3.10)
project(cppcheatsheet)
set(CMAKE_CXX_STANDARD 23)

option(ENABLE_RDMA "Enable RDMA/EFA support" OFF)
option(ENABLE_CUDA "Enable CUDA support" OFF)

include(cmake/packages.cmake)

if(ENABLE_RDMA)
  include_directories(/opt/amazon/efa/include)
  link_directories(/opt/amazon/efa/lib)
endif()

if(ENABLE_CUDA)
  include_directories(/usr/local/cuda/include)
  link_directories(/usr/local/cuda/lib64 /opt/nccl/build/lib)
endif()

add_subdirectory(src)
