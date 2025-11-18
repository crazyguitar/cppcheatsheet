================
External Project
================

.. contents:: Table of Contents
    :backlinks: none


Add an External Project
-----------------------

.. code-block:: cmake

    include (ExternalProject)
    ExternalProject_Add(fmt
      GIT_REPOSITORY "https://github.com/fmtlib/fmt.git"
      GIT_TAG "7.1.3"
      GIT_CONFIG advice.detachedHead=false
      PREFIX "${CMAKE_BINARY_DIR}/fmt"
      CMAKE_CACHE_ARGS
        "-DFMT_INSTALL:BOOL=ON"
        "-DFMT_DOC:BOOL=OFF"
        "-DFMT_TEST:BOOL=OFF"
        "-DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_BINARY_DIR}"
    )

Download Only
-------------

.. code-block:: cmake

    include (ExternalProject)
    ExternalProject_Add(fmt
      GIT_REPOSITORY "https://github.com/fmtlib/fmt.git"
      GIT_TAG "7.1.3"
      GIT_CONFIG advice.detachedHead=false
      PREFIX "${CMAKE_BINARY_DIR}/fmt"
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
    )

Build via GNU Autotool
----------------------

.. code-block:: cmake

    include (ExternalProject)
    ExternalProject_Add(curl
      URL "https://github.com/curl/curl/releases/download/curl-7_74_0/curl-7.74.0.tar.gz"
      URL_MD5 "45f468aa42c4af027c4c6ddba58267f0" # md5sum curl_7.74.0.tar.gz
      BUILD_IN_SOURCE 1
      SOURCE_DIR ${CMAKE_BINARY_DIR}/curl
      CONFIGURE_COMMAND ${CMAKE_BINARY_DIR}/curl/configure --prefix=${CMAKE_BINARY_DIR}
      BUILD_COMMAND make
      INSTALL_COMMAND make install
    )
