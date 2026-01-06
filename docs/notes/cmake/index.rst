CMake
=====

.. meta::
   :description: CMake build system tutorial covering project configuration, dependency management, package integration, and cross-platform C/C++ compilation.
   :keywords: CMake, build system, C++ build, cross-platform compilation, find_package, ExternalProject, dependency management, CMakeLists.txt

Modern C and C++ projects demand build systems that scale across platforms, manage dependencies
gracefully, and integrate with diverse toolchains. CMake has emerged as the de facto standard,
generating native build files for Make, Ninja, Visual Studio, and Xcode from a single
configuration.

From basic project setup to advanced dependency management, these guides cover CMakeLists.txt
fundamentals, the find_package mechanism for locating libraries, and ExternalProject/FetchContent
modules for incorporating third-party code. Whether targeting embedded systems or cloud
infrastructure, CMake provides the flexibility to build anywhere.

.. toctree::
   :maxdepth: 1

   cmake_basic
   cmake_package
   cmake_external
