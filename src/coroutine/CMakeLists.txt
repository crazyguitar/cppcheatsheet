# Build io library
add_library(coro_io STATIC io/selector.cc io/handle.cc)
target_include_directories(coro_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(coro_io PUBLIC cxx_std_20)

# Test executables
foreach(dir basics promise-type awaiter generator result scheduler sleep future coro stream echo-server)
  add_subdirectory(${dir})
endforeach()
