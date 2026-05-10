if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.28"
   AND CMAKE_GENERATOR MATCHES "Ninja"
   AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "14")
  # Modules require CMake 3.28+, Ninja generator, and GCC 14+
  set(CMAKE_CXX_SCAN_FOR_MODULES ON)

  # Basic module example
  add_library(math-module)
  target_sources(math-module
    PUBLIC
      FILE_SET CXX_MODULES FILES
        basic/math.cppm
  )
  target_compile_features(math-module PUBLIC cxx_std_20)

  add_executable(modules-basic basic/main.cc)
  target_link_libraries(modules-basic PRIVATE math-module GTest::gtest_main)
  target_compile_features(modules-basic PRIVATE cxx_std_20)
  gtest_discover_tests(modules-basic)

  # Partitions example
  add_library(graphics-module)
  target_sources(graphics-module
    PUBLIC
      FILE_SET CXX_MODULES FILES
        partitions/graphics.cppm
        partitions/graphics-shapes.cppm
        partitions/graphics-colors.cppm
  )
  target_compile_features(graphics-module PUBLIC cxx_std_20)

  add_executable(modules-partitions partitions/main.cc)
  target_link_libraries(modules-partitions PRIVATE graphics-module GTest::gtest_main)
  target_compile_features(modules-partitions PRIVATE cxx_std_20)
  gtest_discover_tests(modules-partitions)
else()
  message(STATUS "Skipping modules examples (requires CMake 3.28+, Ninja, and GCC 14+)")
endif()
