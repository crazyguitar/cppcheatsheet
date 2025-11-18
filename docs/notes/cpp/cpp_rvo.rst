===============================
Return Value Optimization (RVO)
===============================

.. contents:: Table of Contents
    :backlinks: none


Before starting
---------------

.. code-block:: cpp

    // foo.h

    #include <iostream>

    struct Foo {
        Foo() {
            std::cout << "Constructor" << "\n";
        }
        ~Foo() {
            std::cout << "Destructor" << "\n";
        }
        Foo(const Foo&) {
            std::cout << "Copy Constructor" << "\n";
        }
        Foo(Foo &&) {
            std::cout << "Move Constructor" << "\n";
        }
        Foo& operator=(const Foo&) {
            std::cout << "Copy Assignment" << "\n";
            return *this;
        }
        Foo& operator=(Foo &&){
            std::cout << "Move Assignment" << "\n";
            return *this;
        }
    };

Return Value Optimization
-------------------------

.. code-block:: cpp

    #include "foo.h"

    Foo FooRVO() {
        return Foo();
    }

    int main(int argc, char *argv[]) {
        Foo f = FooRVO();
    }


Named Return Value Optimization
-------------------------------

.. code-block:: cpp

    #include "foo.h"

    Foo FooNRVO() {
        Foo foo;
        return foo;
    }

    int main(int argc, char *argv[]) {
        Foo f = FooNRVO();
    }

Copy Elision
------------

.. code-block:: cpp

    #include "foo.h"

    void CopyElision(Foo foo) {}

    int main(int argc, char *argv[]) {
        CopyElision(Foo());
    }

Return a Global (w/o RVO)
-------------------------

.. code-block:: cpp

    #include "foo.h"

    const Foo foo;

    Foo ReturnGlobal() {
        return foo;
    }

    int main(int argc, char *argv[]) {
        Foo f = ReturnGlobal();
    }

Return a Parameter (w/o RVO)
----------------------------

.. code-block:: cpp

    #include "foo.h"

    Foo ReturnParam(Foo foo) {
        return foo;
    }

    int main(int argc, char *argv[]) {
        Foo f = ReturnParam(Foo());
    }

Runtime Decision (w/ RVO)
-------------------------

.. code-block:: cpp

    #include "foo.h"

    Foo FooRVO(bool is_x) {
        return is_x ? Foo() : Foo();
    }

    int main(int argc, char *argv[]) {
        Foo foo = FooRVO(true);
    }

Runtime Decision (w/ RVO, w/o NRVO)
-----------------------------------

.. code-block:: cpp

    #include "foo.h"

    Foo RVOButNoNRVO(bool is_x) {
        Foo x;
        return is_x ? x : Foo();
    }

    int main(int argc, char *argv[]) {
        Foo f = RVOButNoNRVO(false);
    }

Runtime Decision (w/o NRVO)
---------------------------

.. code-block:: cpp

    #include "foo.h"

    Foo FooNoNRVO(bool is_x) {
        Foo x, y;
        return is_x ? x : y;
    }

    int main(int argc, char *argv[]) {
        Foo foo = FooNoNRVO(true);
    }

Return by ``std::move`` (w/o RVO)
---------------------------------

.. code-block:: cpp

    #include "foo.h"

    #include <utility>

    Foo FooMove() {
        return std::move(Foo());
    }

    int main(int argc, char *argv[]) {
        Foo foo = FooMove();
    }

Return by ``std::move`` (w/o NRVO)
----------------------------------

.. code-block:: cpp

    #include "foo.h"

    #include <utility>

    Foo FooMove() {
        Foo foo;
        return std::move(foo);
    }

    int main(int argc, char *argv[]) {
        Foo foo = FooMove();
    }

Return a Member (w/o RVO)
-------------------------

.. code-block:: cpp

    #include "foo.h"

    struct Bar {
        Foo foo;
    };

    Foo ReturnMember() {
        return Bar().foo;
    }

    int main(int argc, char *argv[]) {
        Foo f = ReturnMember();
    }
