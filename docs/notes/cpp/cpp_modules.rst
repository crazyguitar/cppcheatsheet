=======
Modules
=======

.. meta::
   :description: C++20 modules tutorial covering module declarations, exports, imports, partitions, and migration from headers. Learn how modules improve compilation speed and code organization.
   :keywords: C++20, modules, import, export, module partition, header units, compilation speed, modern C++

.. contents:: Table of Contents
    :backlinks: none

C++20 modules replace the traditional preprocessor-based ``#include`` system with
a modern import mechanism. Unlike headers that are textually included and re-parsed
for every translation unit, modules are compiled once into a binary format and
imported efficiently. This eliminates redundant parsing, prevents macro leakage
between files, removes include order dependencies, and significantly improves
compilation times for large projects.

Module Basics
-------------

:Source: `src/modules/basic <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/modules/basic>`_

A module consists of a module interface unit (the public API) and optionally
module implementation units. The ``export`` keyword marks declarations visible
to importers. Everything not exported remains internal to the module.

.. code-block:: cpp

    // math.cppm - module interface unit
    export module math;  // module declaration

    // Exported - visible to importers
    export int add(int a, int b) {
      return a + b;
    }

    export int multiply(int a, int b) {
      return a * b;
    }

    // Not exported - internal to module
    int helper() {
      return 42;
    }

.. code-block:: cpp

    // main.cpp
    import math;  // import the module

    int main() {
      int sum = add(1, 2);       // OK: exported
      int prod = multiply(3, 4); // OK: exported
      // helper();               // Error: not exported
    }

Export Declarations
-------------------

Multiple ways to export declarations from a module. Use export blocks to group
related exports together, or export individual declarations for fine-grained
control over the public interface.

.. code-block:: cpp

    export module shapes;

    // Export individual declaration
    export constexpr double pi = 3.14159265359;

    // Export a class
    export class Circle {
     public:
      Circle(double r) : radius(r) {}
      double area() const { return pi * radius * radius; }

     private:
      double radius;
    };

    // Export block - everything inside is exported
    export {
      class Rectangle {
       public:
        Rectangle(double w, double h) : width(w), height(h) {}
        double area() const { return width * height; }

       private:
        double width, height;
      };

      double triangle_area(double base, double height) {
        return 0.5 * base * height;
      }
    }

    // Export namespace
    export namespace geometry {
      double circumference(double radius) {
        return 2 * pi * radius;
      }
    }

Module Implementation Units
---------------------------

Separate interface from implementation by using module implementation units.
The implementation unit has access to all module internals but cannot export
new declarations. This pattern keeps interfaces clean and reduces recompilation
when only implementation changes.

.. code-block:: cpp

    // calculator.cppm - interface unit
    export module calculator;

    export class Calculator {
     public:
      int add(int a, int b);
      int subtract(int a, int b);
      int multiply(int a, int b);
      int divide(int a, int b);
    };

.. code-block:: cpp

    // calculator_impl.cpp - implementation unit
    module calculator;  // no 'export' keyword

    int Calculator::add(int a, int b) {
      return a + b;
    }

    int Calculator::subtract(int a, int b) {
      return a - b;
    }

    int Calculator::multiply(int a, int b) {
      return a * b;
    }

    int Calculator::divide(int a, int b) {
      return b != 0 ? a / b : 0;
    }

Module Partitions
-----------------

:Source: `src/modules/partitions <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/modules/partitions>`_

Large modules can be split into partitions for better organization. Partitions
are internal to the module and must be explicitly re-exported from the primary
module interface if they should be visible to importers.

.. code-block:: cpp

    // graphics-shapes.cppm - partition
    export module graphics:shapes;

    export class Point {
     public:
      double x, y;
    };

    export class Line {
     public:
      Point start, end;
    };

.. code-block:: cpp

    // graphics-colors.cppm - partition
    export module graphics:colors;

    export struct Color {
      unsigned char r, g, b, a;
    };

    export constexpr Color Red{255, 0, 0, 255};
    export constexpr Color Green{0, 255, 0, 255};
    export constexpr Color Blue{0, 0, 255, 255};

.. code-block:: cpp

    // graphics.cppm - primary interface
    export module graphics;

    export import :shapes;  // re-export shapes partition
    export import :colors;  // re-export colors partition

    export class Canvas {
     public:
      void draw_line(const Line& line, const Color& color);
    };

.. code-block:: cpp

    // main.cpp
    import graphics;

    int main() {
      Point p1{0, 0};
      Point p2{100, 100};
      Line line{p1, p2};

      Canvas canvas;
      canvas.draw_line(line, Red);
    }

Importing Standard Library
--------------------------

C++23 introduces ``import std;`` to import the entire standard library as a module.
For C++20, you can import individual standard headers as header units, though
compiler support varies.

.. code-block:: cpp

    // C++23: import entire standard library
    import std;

    int main() {
      std::vector<std::string> names{"Alice", "Bob"};
      for (const auto& name : names) {
        std::cout << name << "\n";
      }
    }

.. code-block:: cpp

    // C++20: import standard headers as header units (compiler-dependent)
    import <vector>;
    import <string>;
    import <iostream>;

    int main() {
      std::vector<int> v{1, 2, 3};
      std::cout << v.size() << "\n";
    }

Global Module Fragment
----------------------

The global module fragment allows including traditional headers that must be
processed by the preprocessor before the module declaration. Use this for
headers that rely on macros or cannot be imported as header units.

.. code-block:: cpp

    module;  // start global module fragment

    // Traditional includes processed here
    #include <cstddef>
    #include <cassert>
    #ifdef _WIN32
    #include <windows.h>
    #endif

    export module platform;  // module declaration ends global fragment

    export size_t get_page_size() {
    #ifdef _WIN32
      SYSTEM_INFO si;
      GetSystemInfo(&si);
      return si.dwPageSize;
    #else
      return 4096;
    #endif
    }

Building Modules with CMake
---------------------------

CMake 3.28+ supports C++20 modules. Enable with ``CXX_SCAN_FOR_MODULES`` and
use ``FILE_SET`` to specify module interface files.

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.28)
    project(myproject CXX)

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    add_library(math)
    target_sources(math
      PUBLIC
        FILE_SET CXX_MODULES FILES
          math.cppm
    )

    add_executable(main main.cpp)
    target_link_libraries(main PRIVATE math)

Compiler Support
----------------

Module support varies across compilers. Check your compiler version and flags:

+----------+----------------+----------------------------------+
| Compiler | Min Version    | Flags                            |
+==========+================+==================================+
| MSVC     | VS 2019 16.10+ | ``/std:c++20`` (best support)    |
+----------+----------------+----------------------------------+
| GCC      | 11+            | ``-std=c++20 -fmodules-ts``      |
+----------+----------------+----------------------------------+
| Clang    | 16+            | ``-std=c++20 -fmodules``         |
+----------+----------------+----------------------------------+

See Also
--------

- :doc:`cpp_cmake` - CMake build system
- :doc:`cpp_basic` - C++ fundamentals
