=======
Package
=======

.. contents:: Table of Contents
    :backlinks: none

Support ``find_package``
------------------------

.. code-block:: cmake

    # fib
    # ├── CMakeLists.txt
    # ├── cmake
    # │   └── FibConfig.cmake
    # ├── include
    # │   └── fib
    # │       └── fib.h
    # └── src
    #     └── fib.cc

    cmake_minimum_required(VERSION 3.10)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
    project(example)

    set(target fib)
    add_library(${target} src/fib.cc)
    target_compile_options(${target} PRIVATE -Wall -Werror -Wextra)
    target_include_directories(${target}
      PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
      PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    include(GNUInstallDirs)
    install(TARGETS ${target} EXPORT FibTargets
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib
      RUNTIME DESTINATION bin
      INCLUDES DESTINATION include
    )

    install(DIRECTORY include/fib
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(EXPORT FibTargets
      FILE FibTargets.cmake
      NAMESPACE Fib::
      DESTINATION lib/cmake/Fib
    )
    include(CMakePackageConfigHelpers)
    write_basic_package_version_file("FibConfigVersion.cmake"
      VERSION 1.0.0
      COMPATIBILITY SameMajorVersion
    )
    install(FILES "cmake/FibConfig.cmake"
      "${PROJECT_BINARY_DIR}/FibConfigVersion.cmake"
      DESTINATION lib/cmake/Fib
    )

cmake/FibConfig.cmake

.. code-block:: cmake

    include(CMakeFindDependencyMacro)
    find_dependency(Boost)
    include("${CMAKE_CURRENT_LIST_DIR}/FibTargets.cmake")

CPack
-----

.. code-block:: cmake

    # $ cd build
    # $ cmake ..
    # $ make -j 2
    # $ cpack -G TGZ .

    cmake_minimum_required(VERSION 3.10)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
    project(a)

    add_executable(a a.cc)
    add_library(b b.cc)
    target_link_libraries(a PRIVATE b)
    include(GNUInstallDirs)
    install(TARGETS a b
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
    install(FILES b.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    set(CPACK_GENERATOR "ZIP;TGZ")
    SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "crazyguitar")
    include(CPack)
