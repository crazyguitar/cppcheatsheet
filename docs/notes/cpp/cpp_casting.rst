=======
Casting
=======

.. meta::
   :description: C++ type casting guide covering static_cast, dynamic_cast, const_cast, reinterpret_cast with examples and best practices.
   :keywords: C++, casting, static_cast, dynamic_cast, const_cast, reinterpret_cast, type conversion

.. contents:: Table of Contents
    :backlinks: none

C-Style Cast (Legacy Casting)
-----------------------------

:Source: `src/casting/c-style <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/casting/c-style>`_

C-style casts use the syntax ``(type)expression`` inherited from C. While
concise, they are dangerous because the compiler tries ``const_cast``,
``static_cast``, and ``reinterpret_cast`` in sequence until one succeeds.
This makes it hard to know which conversion actually happens. The lack of
visual distinction also makes C-style casts difficult to search for in
large codebases. Modern C++ code should prefer explicit C++ casts for
clarity, safety, and maintainability.

.. code-block:: cpp

    #include <cmath>
    #include <iostream>

    int main(int argc, char *argv[]) {
      double x = M_PI;
      int xx = (int)x;  // Truncates: 3.14159... -> 3
      std::cout << xx << "\n";

      long z = LONG_MAX;
      int zz = (int)z;  // Dangerous: overflow, undefined behavior
      std::cout << zz << "\n";
    }

.. warning::

    C-style casts bypass type safety checks and can silently perform dangerous
    conversions. Prefer ``static_cast`` for safe conversions and avoid C-style
    casts in modern C++ code.

const_cast: Removing or Adding const
------------------------------------

:Source: `src/casting/const-cast <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/casting/const-cast>`_

The ``const_cast`` operator adds or removes ``const`` (or ``volatile``)
qualifiers from a pointer or reference. This is the only C++ cast that can
modify cv-qualifiers. The primary use case is interfacing with legacy C APIs
or older C++ code that lacks proper const-correctness. Modifying a truly
const object through ``const_cast`` results in undefined behavior, so it
should only be used when you are certain the underlying object was not
originally declared as const.

.. code-block:: cpp

    #include <iostream>

    void legacy_api(int &x) {
      const_cast<int &>(x) = 0;  // Remove const to modify
    }

    int main(int argc, char *argv[]) {
      int x = 123;
      legacy_api(x);
      std::cout << x << "\n";  // Output: 0
    }

.. note::

    Using ``const_cast`` to modify an object that was originally declared
    ``const`` is undefined behavior. Only use it when you know the underlying
    object is non-const.

reinterpret_cast: Low-Level Bit Reinterpretation
------------------------------------------------

:Source: `src/casting/reinterpret-cast <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/casting/reinterpret-cast>`_

The ``reinterpret_cast`` operator performs low-level reinterpretation of
bit patterns between unrelated types. It is primarily used for converting
between pointer types, pointer-to-integer conversions, and accessing raw
memory representations. Common use cases include serialization, memory-mapped
I/O, and interfacing with hardware. This cast provides no type safety and
the results are implementation-defined, so it should be used sparingly and
only in systems programming contexts where such low-level access is necessary.

.. code-block:: cpp

    #include <iostream>

    struct Point {
      int x;
      int y;
    };

    int main(int argc, char *argv[]) {
      Point p{1, 2};

      // View struct as raw bytes
      char *buf = reinterpret_cast<char *>(&p);
      for (size_t i = 0; i < sizeof(Point); ++i) {
        std::cout << static_cast<int>(buf[i]) << " ";
      }
      // Output (little-endian): 1 0 0 0 2 0 0 0
    }

static_cast: Compile-Time Type Conversion
-----------------------------------------

:Source: `src/casting/static-cast <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/casting/static-cast>`_

The ``static_cast`` operator performs well-defined conversions between
related types at compile time. It handles numeric conversions (int to double),
enum-to-int conversions, void pointer conversions, and pointer upcasts in
class hierarchies. It can also invoke explicit constructors and conversion
operators. Unlike ``dynamic_cast``, it performs no runtime type checking,
so downcasting with ``static_cast`` is unsafe unless you are certain of the
object's actual type. When the conversion is valid, ``static_cast`` generates
efficient code with no runtime overhead.

.. code-block:: cpp

    #include <iostream>
    #include <memory>

    struct Base {
      virtual ~Base() = default;
    };

    struct Derived : public Base {
      int value = 42;
    };

    int main(int argc, char *argv[]) {
      auto d = std::make_unique<Derived>();

      // Safe upcast: Derived* -> Base*
      Base *b = static_cast<Base *>(d.get());

      // Unsafe downcast: only safe if b actually points to Derived
      auto d2 = static_cast<Derived *>(b);
      std::cout << d2->value << "\n";  // Output: 42
    }

.. warning::

    Using ``static_cast`` for downcasting is unsafe. If the object is not
    actually of the target type, the behavior is undefined. Use ``dynamic_cast``
    when runtime type checking is needed.

dynamic_cast: Runtime Type-Safe Downcasting
-------------------------------------------

:Source: `src/casting/dynamic-cast <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/casting/dynamic-cast>`_

The ``dynamic_cast`` operator performs runtime type checking for safe
downcasting in polymorphic class hierarchies. It requires the source type
to have at least one virtual function (typically the destructor) because
it relies on RTTI (Run-Time Type Information) stored in the virtual table.
For pointer casts, it returns ``nullptr`` on failure; for reference casts,
it throws ``std::bad_cast``. This is the safest way to navigate class
hierarchies when the actual type is uncertain at compile time, such as when
processing heterogeneous collections of objects.

.. code-block:: cpp

    #include <iostream>
    #include <memory>

    struct Base {
      virtual ~Base() = default;
    };

    struct Derived : public Base {
      void greet() { std::cout << "Hello from Derived\n"; }
    };

    int main(int argc, char *argv[]) {
      auto base = std::make_unique<Base>();
      auto derived = std::make_unique<Derived>();

      // Failed downcast: base doesn't point to Derived
      auto d1 = dynamic_cast<Derived *>(base.get());
      std::cout << "base -> Derived: " << (d1 ? "success" : "failed") << "\n";

      // Successful upcast: Derived* -> Base*
      auto b2 = dynamic_cast<Base *>(derived.get());
      std::cout << "Derived -> Base: " << (b2 ? "success" : "failed") << "\n";
    }

.. note::

    ``dynamic_cast`` has runtime overhead due to RTTI (Run-Time Type Information).
    In performance-critical code where types are known, prefer ``static_cast``.
