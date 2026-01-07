=============================
Preprocessor & GNU Extensions
=============================

.. meta::
   :description: C preprocessor macros and GNU C extensions covering predefined macros, variadic macros, statement expressions, typeof, attribute specifiers, and compiler builtins.
   :keywords: C preprocessor, C macros, GNU C extensions, __attribute__, typeof, statement expressions, variadic macros, __VA_ARGS__, __builtin, GCC extensions

.. contents:: Table of Contents
    :backlinks: none

Introduction
------------

The C preprocessor transforms source code before compilation, enabling conditional
compilation, macro expansion, and file inclusion. Beyond standard preprocessor
features, GCC provides powerful extensions like statement expressions, typeof,
and attribute specifiers that enable safer and more expressive code. While these
extensions reduce portability, they're widely supported by Clang and essential
for systems programming, especially in the Linux kernel.

Predefined Macros
-----------------

:Source: `src/c/predefined-macros <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/predefined-macros>`_

The C preprocessor automatically defines several macros that provide compile-time
information about the source file, compilation environment, and current location
in the code. These predefined macros are invaluable for debugging, logging, and
generating diagnostic messages that include file names, line numbers, and function
names. They enable creating informative error messages without hardcoding location
information, and support conditional compilation based on compiler features and
standards compliance.

.. code-block:: c

    #include <stdio.h>

    int main(void) {
      printf("File: %s\n", __FILE__);
      printf("Date: %s\n", __DATE__);
      printf("Time: %s\n", __TIME__);
      printf("Line: %d\n", __LINE__);
      printf("Func: %s\n", __func__);
    }

.. code-block:: console

    $ ./predefined
    File: main.c
    Date: Jan 06 2026
    Time: 10:30:00
    Line: 7
    Func: main

Common predefined macros:

.. code-block:: text

    __FILE__       Current source file name
    __LINE__       Current line number
    __func__       Current function name (C99)
    __DATE__       Compilation date "MMM DD YYYY"
    __TIME__       Compilation time "HH:MM:SS"
    __STDC__       1 if compiler conforms to ISO C
    __GNUC__       GCC major version number

Conditional Compilation
-----------------------

Conditional compilation directives enable including or excluding code based on
compile-time conditions, making it possible to maintain a single codebase that
adapts to different platforms, configurations, and build types. The ``#ifdef``
and ``#if defined()`` directives test whether macros are defined, while ``#if``
evaluates constant expressions. This mechanism is essential for platform-specific
code, feature flags, debug builds, and header guards. The ``-D`` compiler flag
defines macros from the command line, enabling build-time configuration without
modifying source files.

.. code-block:: c

    #include <stdio.h>

    int main(void) {
    #ifdef DEBUG
      printf("Debug build\n");
    #else
      printf("Release build\n");
    #endif

    #if defined(__linux__)
      printf("Linux\n");
    #elif defined(__APPLE__)
      printf("macOS\n");
    #endif
    }

.. code-block:: console

    $ gcc -o test main.c && ./test
    Release build
    $ gcc -DDEBUG -o test main.c && ./test
    Debug build

Variadic Macros
---------------

:Source: `src/c/variadic-macros <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/variadic-macros>`_

Variadic macros accept a variable number of arguments using the ellipsis (``...``)
syntax, with arguments accessible through the special ``__VA_ARGS__`` identifier.
This enables creating flexible logging, debugging, and wrapper macros that mirror
the interface of functions like ``printf``. The GNU extension ``##__VA_ARGS__``
provides a crucial enhancement: it removes the preceding comma when no variadic
arguments are provided, allowing macros to work correctly with both zero and
multiple optional arguments. This pattern is ubiquitous in logging frameworks
and debug utilities.

.. code-block:: c

    #include <stdio.h>

    #define LOG(fmt, ...) \
        printf("[LOG] " fmt "\n", ##__VA_ARGS__)

    #define DEBUG_LOG(fmt, ...) \
        printf("[%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)

    int main(void) {
      LOG("Starting");
      LOG("Value: %d", 42);
      DEBUG_LOG("x = %d, y = %d", 10, 20);
    }

.. code-block:: console

    $ ./variadic
    [LOG] Starting
    [LOG] Value: 42
    [main.c:12] x = 10, y = 20

Array Size Macro
----------------

:Source: `src/c/arraysize <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/arraysize>`_

Calculating array size at compile time is a fundamental C idiom that divides the
total array size by the size of a single element. This technique only works for
actual arrays with known size at compile time—it fails silently for pointers,
which is a common source of bugs. The macro approach is safer than manually
tracking array sizes and automatically updates when elements are added or removed.
Modern C++ provides ``std::size()`` for this purpose, but in C, this macro pattern
remains the standard solution.

.. code-block:: c

    #include <stdio.h>

    #define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

    int main(void) {
      int nums[] = {1, 2, 3, 4, 5};
      char *strs[] = {"hello", "world", NULL};

      printf("nums size: %zu\n", ARRAY_SIZE(nums));
      printf("strs size: %zu\n", ARRAY_SIZE(strs));
    }

.. code-block:: console

    $ ./arraysize
    nums size: 5
    strs size: 3

Statement Expressions (GNU)
---------------------------

:Source: `src/c/stmt-expr <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/stmt-expr>`_

Statement expressions are a GNU extension that allows compound statements (blocks)
to appear within expressions, enclosed in ``({ })``. The value of the last
statement becomes the value of the entire expression. This enables macros to
declare local variables, execute multiple statements, and return a computed value
—all while evaluating arguments only once. This solves the classic macro problem
where ``MAX(a++, b++)`` would increment arguments multiple times. Statement
expressions are heavily used in the Linux kernel and other systems code for
creating safe, type-generic macros.

.. code-block:: c

    #include <stdio.h>

    #define max(a, b) ({   \
        typeof(a) _a = (a); \
        typeof(b) _b = (b); \
        _a > _b ? _a : _b;  \
    })

    #define swap(a, b) ({   \
        typeof(a) _tmp = a; \
        a = b;              \
        b = _tmp;           \
    })

    int main(void) {
      int x = 10, y = 20;
      printf("max(%d, %d) = %d\n", x, y, max(x, y));

      swap(x, y);
      printf("after swap: x=%d, y=%d\n", x, y);
    }

.. code-block:: console

    $ ./stmt-expr
    max(10, 20) = 20
    after swap: x=20, y=10

``typeof`` Operator (GNU)
-------------------------

:Source: `src/c/typeof-gnu <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/typeof-gnu>`_

The ``typeof`` operator is a GNU extension that returns the type of an expression
at compile time, enabling the creation of type-safe generic macros. Unlike C++
templates, ``typeof`` works purely at the preprocessor/compiler level, allowing
macros to automatically adapt to any data type passed to them. This eliminates
the need for type-specific macro variants and prevents subtle bugs from type
mismatches. C23 standardizes this feature, making it portable across compilers.

.. code-block:: c

    #include <stdio.h>

    #define SWAP(a, b) do {   \
        typeof(a) _tmp = (a); \
        (a) = (b);            \
        (b) = _tmp;           \
    } while (0)

    #define MIN(a, b) ({       \
        typeof(a) _a = (a);    \
        typeof(b) _b = (b);    \
        _a < _b ? _a : _b;     \
    })

    int main(void) {
      int i = 5, j = 10;
      double x = 3.14, y = 2.71;

      SWAP(i, j);
      printf("i=%d, j=%d\n", i, j);
      printf("min(%.2f, %.2f) = %.2f\n", x, y, MIN(x, y));
    }

.. code-block:: console

    $ ./typeof
    i=10, j=5
    min(3.14, 2.71) = 2.71

``__attribute__`` Specifiers (GNU)
----------------------------------

:Source: `src/c/attribute <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/attribute>`_

GCC attributes provide compiler hints for optimization, diagnostics, and code
generation through the ``__attribute__`` syntax. These annotations help the
compiler generate better code, catch bugs at compile time, and control low-level
details like struct layout and function calling conventions. Common use cases
include ``packed`` for removing struct padding in network protocols, ``unused``
to suppress warnings for intentionally unused variables, ``format`` for printf-style
argument checking, and ``noreturn`` for functions that never return. While
non-portable, most attributes are supported by Clang and are essential for
systems programming.

.. code-block:: c

    #include <stdio.h>
    #include <stdint.h>

    struct __attribute__((packed)) packet {
      uint8_t type;
      uint32_t data;  // no padding
    };

    __attribute__((unused))
    static void helper(void) { }

    __attribute__((format(printf, 1, 2)))
    void log_msg(const char *fmt, ...) {
      va_list args;
      va_start(args, fmt);
      vprintf(fmt, args);
      va_end(args);
    }

    __attribute__((noreturn))
    void die(const char *msg) {
      fprintf(stderr, "Fatal: %s\n", msg);
      exit(1);
    }

    int main(void) {
      printf("packed struct size: %zu\n", sizeof(struct packet));
    }

.. code-block:: console

    $ ./attribute
    packed struct size: 5

Common attributes:

.. code-block:: text

    __attribute__((packed))           Remove struct padding
    __attribute__((aligned(n)))       Align to n bytes
    __attribute__((unused))           Suppress unused warnings
    __attribute__((deprecated))       Mark as deprecated
    __attribute__((noreturn))         Function never returns
    __attribute__((format(printf,m,n))) Check printf format
    __attribute__((constructor))      Run before main()
    __attribute__((destructor))       Run after main()

Constructor and Destructor
--------------------------

:Source: `src/c/ctor-dtor <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/ctor-dtor>`_

Functions marked with ``__attribute__((constructor))`` execute automatically
before ``main()`` begins, while ``__attribute__((destructor))`` functions run
after ``main()`` returns or ``exit()`` is called. This mechanism enables automatic
initialization and cleanup without explicit calls, similar to C++ static object
constructors. Common use cases include initializing global state, registering
signal handlers, setting up logging systems, and releasing resources. Multiple
constructors/destructors can specify priority values to control execution order.

.. code-block:: c

    #include <stdio.h>

    __attribute__((constructor))
    void init(void) {
      printf("Initializing...\n");
    }

    __attribute__((destructor))
    void cleanup(void) {
      printf("Cleaning up...\n");
    }

    int main(void) {
      printf("In main()\n");
      return 0;
    }

.. code-block:: console

    $ ./ctor-dtor
    Initializing...
    In main()
    Cleaning up...

Compiler Builtins
-----------------

:Source: `src/c/builtins <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/builtins>`_

GCC provides builtin functions that map directly to efficient machine instructions
or provide compiler-specific functionality not available in standard C. Bit
manipulation builtins like ``__builtin_popcount`` and ``__builtin_clz`` compile
to single CPU instructions on modern processors, offering significant performance
gains over manual implementations. Branch prediction hints via ``__builtin_expect``
help the compiler optimize hot paths, while ``__builtin_unreachable`` enables
dead code elimination. These builtins are widely supported by Clang and are
commonly used in performance-critical code and operating system kernels.

.. code-block:: c

    #include <stdio.h>

    int main(void) {
      unsigned int x = 0b10110000;

      printf("popcount(0x%x) = %d\n", x, __builtin_popcount(x));
      printf("clz(0x%x) = %d\n", x, __builtin_clz(x));
      printf("ctz(0x%x) = %d\n", x, __builtin_ctz(x));
      printf("ffs(0x%x) = %d\n", x, __builtin_ffs(x));

      // Branch prediction hints
      int likely_true = 1;
      if (__builtin_expect(likely_true, 1)) {
        printf("Expected path\n");
      }
    }

.. code-block:: console

    $ ./builtins
    popcount(0xb0) = 4
    clz(0xb0) = 24
    ctz(0xb0) = 4
    ffs(0xb0) = 5
    Expected path

Common builtins:

.. code-block:: text

    __builtin_popcount(x)     Count set bits
    __builtin_clz(x)          Count leading zeros
    __builtin_ctz(x)          Count trailing zeros
    __builtin_ffs(x)          Find first set bit (1-indexed)
    __builtin_expect(x, v)    Branch prediction hint
    __builtin_unreachable()   Mark unreachable code
    __builtin_prefetch(p)     Prefetch memory

Overflow Checking Builtins
--------------------------

:Source: `src/c/overflow-check <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/overflow-check>`_

Integer overflow in C is undefined behavior for signed types and wraps silently
for unsigned types, making it a common source of security vulnerabilities. GCC's
overflow checking builtins perform arithmetic operations and return a boolean
indicating whether overflow occurred, enabling safe arithmetic without undefined
behavior. These functions are essential for security-critical code handling
untrusted input, memory allocation size calculations, and any arithmetic that
could exceed type limits. The result is stored via pointer, allowing both the
overflow check and the computed value to be used.

.. code-block:: c

    #include <stdio.h>
    #include <stdint.h>

    int main(void) {
      int a = INT32_MAX, b = 1, result;

      if (__builtin_add_overflow(a, b, &result)) {
        printf("Overflow detected!\n");
      } else {
        printf("Result: %d\n", result);
      }

      // Safe multiplication
      size_t count = 1000000, size = 10000, total;
      if (__builtin_mul_overflow(count, size, &total)) {
        printf("Allocation would overflow\n");
      }
    }

.. code-block:: console

    $ ./overflow
    Overflow detected!
    Allocation would overflow

Binary Constants (GNU)
----------------------

:Source: `src/c/binary-const <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/binary-const>`_

Binary literals with the ``0b`` or ``0B`` prefix allow expressing integer constants
in base-2 notation, making bit patterns immediately readable without mental
conversion from hexadecimal. This is particularly valuable when working with
hardware registers, protocol flags, bitmasks, and any code where individual bit
positions have specific meanings. Originally a GNU extension, binary constants
are now standardized in C23, ensuring portability across modern compilers while
maintaining backward compatibility with GCC and Clang.

.. code-block:: c

    #include <stdio.h>

    #define BIT(n) (1U << (n))

    int main(void) {
      int flags = 0b10110100;
      int mask  = 0b00001111;

      printf("flags = 0x%02x\n", flags);
      printf("flags & mask = 0x%02x\n", flags & mask);
      printf("BIT(3) = 0x%02x\n", BIT(3));
    }

.. code-block:: console

    $ ./binary
    flags = 0xb4
    flags & mask = 0x04
    BIT(3) = 0x08

Nested Functions (GNU)
----------------------

:Source: `src/c/nested-func <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/nested-func>`_

GCC allows defining functions inside other functions, creating nested functions
with access to the enclosing function's local variables through lexical scoping.
This provides closure-like behavior in C, where the inner function captures and
can modify variables from its enclosing scope. Nested functions are useful for
creating helper functions that need access to local state without passing many
parameters or using global variables. However, taking the address of a nested
function requires executable stack trampolines, which may conflict with security
hardening measures on some systems.

.. code-block:: c

    #include <stdio.h>

    int main(void) {
      int multiplier = 10;

      int multiply(int x) {
        return x * multiplier;  // accesses outer variable
      }

      printf("multiply(5) = %d\n", multiply(5));

      multiplier = 20;
      printf("multiply(5) = %d\n", multiply(5));
    }

.. code-block:: console

    $ ./nested
    multiply(5) = 50
    multiply(5) = 100

Zero-Length Arrays (GNU)
------------------------

:Source: `src/c/flex-array <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/flex-array>`_

Zero-length arrays (``char data[0]``) at the end of structs are a GNU extension
that predates C99 flexible array members. They enable variable-length structures
where the array size is determined at runtime through dynamic allocation. This
pattern is common in network protocols, file formats, and any data structure
where a header is followed by variable-length payload. C99 standardized this
concept as flexible array members (``char data[]``), which should be preferred
in new code for portability. Both approaches require careful memory management
to allocate sufficient space for the actual data.

.. code-block:: c

    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>

    struct old_style {
      int length;
      char data[0];  // GNU zero-length array
    };

    struct new_style {
      int length;
      char data[];   // C99 flexible array member (preferred)
    };

    int main(void) {
      const char *msg = "Hello";
      int len = strlen(msg);

      struct new_style *s = malloc(sizeof(*s) + len + 1);
      s->length = len;
      strcpy(s->data, msg);

      printf("length=%d, data=%s\n", s->length, s->data);
      free(s);
    }

.. code-block:: console

    $ ./flex-array
    length=5, data=Hello

Case Ranges (GNU)
-----------------

:Source: `src/c/case-range <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/case-range>`_

GCC allows ranges in case labels using ``...`` syntax, reducing repetitive code.

.. code-block:: c

    #include <stdio.h>

    const char *grade(int score) {
      switch (score) {
        case 90 ... 100: return "A";
        case 80 ... 89:  return "B";
        case 70 ... 79:  return "C";
        case 60 ... 69:  return "D";
        case 0 ... 59:   return "F";
        default:         return "Invalid";
      }
    }

    int main(void) {
      printf("95 -> %s\n", grade(95));
      printf("73 -> %s\n", grade(73));
      printf("45 -> %s\n", grade(45));
    }

.. code-block:: console

    $ ./case-range
    95 -> A
    73 -> C
    45 -> F
