include(FetchContent)

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.15.1
)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
)

FetchContent_MakeAvailable(spdlog)
FetchContent_MakeAvailable(googletest)

# Common dependencies
find_package(Threads REQUIRED)
find_package(spdlog REQUIRED)

if(ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  find_library(NVML_LIBRARIES nvidia-ml PATHS /usr/local/cuda/lib64/stubs)
  find_library(GDR_LIBRARY gdrapi PATHS /opt/gdrcopy/lib /usr/local/lib /usr/lib)
endif()

if(ENABLE_RDMA)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(HWLOC REQUIRED hwloc)
  find_package(MPI REQUIRED)
endif()

enable_testing()
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)
