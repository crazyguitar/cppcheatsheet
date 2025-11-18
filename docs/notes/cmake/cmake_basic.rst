=====
cmake
=====

.. contents:: Table of Contents
    :backlinks: none

Minimal CMakeLists.txt
----------------------

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.10)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
    project(example)
    add_executable(a.out a.cpp b.cpp)


Wildcard Sourse Files
---------------------

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.10)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
    file(GLOB src "*.cpp")

    project(example)
    add_executable(a.out ${src})

Set CXXFLAGS
------------

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.10)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
    file(GLOB src "*.cc")

    project(example)
    set(CMAKE_CXX_FLAGS "-Wall -Werror -O3")
    add_executable(a.out ${src})

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.10)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
    file(GLOB src "*.cc")

    project(example)
    add_executable(a.out ${src})
    target_compile_options(a.out PRIVATE -Werror)

Set CXXFLAGS with Build Type
----------------------------

.. code-block:: cmake

    # common
    set(CMAKE_CXX_FLAGS "-Wall -Werror -O3")
    # debug
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g")
    # release
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -pedantic")

Build Debug/Release
-------------------

.. code-block:: bash

    $ cmake -DCMAKE_BUILD_TYPE=Release ../
    $ cmake -DCMAKE_BUILD_TYPE=Debug ../

Build with Type
---------------

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.10)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
    project(example)
    add_executable(a.out a.cc)

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      target_compile_options(a.out PRIVATE -g -O0 -Wall)
    else()
      target_compile_options(a.out PRIVATE -O3 -Wall -Werror)
    endif()

Instead of checking ``CMAKE_BUILD_TYPE``, modern CMake prefers to use *generator
expressions* to examine conditions.

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.10)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
    project(example)
    add_executable(a.out a.cc)
    target_compile_options(a.out PRIVATE
      $<IF:$<CONFIG:Debug>, -g -O0 -Wall, -O3 -Wall -Werror>
    )

Version File
------------

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.10)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
    file(GLOB src "*.cpp")

    project(example VERSION 1.0)
    configure_file(version.h.in version.h)

    add_executable(a.out ${src})
    target_include_directories(a.out PUBLIC "${PROJECT_BINARY_DIR}")

version.h.in

.. code-block:: cpp

    #pragma once

    #define VERSION_MAJOR @example_VERSION_MAJOR@
    #define VERSION_MINOR @example_VERSION_MINOR@

Build/Link a static library
---------------------------

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.10)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
    file(GLOB src "*.cpp")

    project(example VERSION 1.0)
    configure_file(version.h.in version.h)

    add_executable(a.out ${src})
    add_library(b b.cpp)
    target_link_libraries(a.out PUBLIC b)
    target_include_directories(a.out PUBLIC "${PROJECT_BINARY_DIR}")

Build/Link a shared library
---------------------------

.. code-block:: cmake


    cmake_minimum_required(VERSION 3.10)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
    file(GLOB src "*.cpp")

    project(example VERSION 1.0)
    configure_file(version.h.in version.h)

    add_executable(a.out ${src})
    add_library(b SHARED b.cpp)
    target_link_libraries(a.out PUBLIC b)
    target_include_directories(a.out PUBLIC "${PROJECT_BINARY_DIR}")

Subdirectory
------------

subdirectory fib/

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.10)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
    file(GLOB src "*.cpp")
    add_library(b SHARED b.cpp)
    target_include_directories(b PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

project dir

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.10)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
    file(GLOB src "*.cpp")

    project(example VERSION 1.0)
    configure_file(version.h.in version.h)

    add_executable(a.out ${src})
    add_subdirectory(fib)
    target_link_libraries(a.out PUBLIC b)
    target_include_directories(a.out PUBLIC
        "${PROJECT_BINARY_DIR}"
        "${PROJECT_BINARY_DIR/fib}"
    )

PUBLIC & PRIVATE
----------------

- PUBLIC - only affect the current target, not dependencies
- INTERFACE - only needed for dependencies

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.10)

    project(example)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
    find_package(Boost)

    add_executable(a.out a.cpp)
    add_library(b STATIC b.cpp b.h)

    target_include_directories(a.out PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_include_directories(b PRIVATE "${Boost_INCLUDE_DIR}")
    target_link_libraries(a.out INTERFACE b) # link b failed

Generator Expression
--------------------

.. code-block:: cmake

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

Install
-------

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.10)
    project(a)
    add_library(b_static STATIC b.cc)
    add_library(b_shared SHARED b.cc)
    add_executable(a a.cc b.cc)

    include(GNUInstallDirs)
    set(INSTALL_TARGETS a b_static b_shared)
    install(TARGETS ${INSTALL_TARGETS}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(FILES b.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

Run a command at configure time
-------------------------------

.. code-block:: cmake

    execute_process(
        COMMAND git submodule update --init --recursive
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE GIT_SUBMOD_RESULT
    )

Option
------

.. code-block:: cmake

    # $ make -p build
    # $ cd build
    # $ cmake -DBUILD_TEST=ON ../

    option(BUILD_TEST "Build test" OFF)
    if (BUILD_TEST)
        message("Build tests.")
    else()
        message("Ignore tests.")
    endif()

Alias a Library
---------------

When a ``CMakeLists.txt`` export Foo in namespace ``Foo::``, it also need to
create an alias ``Foo::Foo``.

.. code-block:: cmake

    add_library(Foo::Foo ALIAS Foo)
