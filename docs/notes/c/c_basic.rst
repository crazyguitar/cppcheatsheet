========
C Basics
========

.. meta::
   :description: Modern C programming fundamentals covering C11, C17, and C23 features including _Generic, _Static_assert, atomics, threads, anonymous structs, and classic C patterns.
   :keywords: C programming, C11, C17, C23, modern C, _Generic, _Static_assert, atomics, stdatomic, threads, anonymous struct, typeof, auto, nullptr, comma operator, callback functions

.. contents:: Table of Contents
    :backlinks: none

Introduction
------------

C continues to evolve with new standards (C11, C17, C23) that add modern features
while maintaining backward compatibility. C11 introduced atomics, threads, and
type-generic expressions. C23 adds ``typeof``, ``auto`` type inference, ``nullptr``,
and improved ``constexpr``. This reference covers both classic C patterns and
modern features essential for systems programming, embedded development, and
high-performance applications.

Comma Operator
--------------

:Source: `src/c/comma-operator <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/comma-operator>`_

The comma operator evaluates its left operand, discards the result, then evaluates
and returns its right operand. This binary operator has the lowest precedence of
all C operators, which often leads to confusion when combined with assignment.
Understanding the distinction between comma as an operator versus comma as a
separator (in function arguments or variable declarations) is crucial for reading
and writing correct C code.

.. code-block:: c

    #include <stdio.h>

    int main(void) {
      int a = 1, b = 2, c = 3;  // comma as separator
      int i;

      i = (a, b, c);      // comma operator: evaluates a, b, returns c
      printf("i = (a, b, c) => %d\n", i);  // prints 3

      i = (a + 5, a + b); // evaluates a+5 (discarded), returns a+b
      printf("i = (a + 5, a + b) => %d\n", i);  // prints 3

      i = a + 5, a + b;   // equivalent to (i = a + 5), a + b
      printf("i = a + 5, a + b => %d\n", i);  // prints 6
    }

.. code-block:: console

    $ ./comma-operator
    i = (a, b, c) => 3
    i = (a + 5, a + b) => 3
    i = a + 5, a + b => 6

Static Variables for State Persistence
--------------------------------------

:Source: `src/c/static-closure <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/static-closure>`_

Static local variables retain their values between function calls, providing a
simple mechanism for maintaining state without global variables. Unlike automatic
(local) variables that are created and destroyed with each function invocation,
static variables are initialized once and persist for the program's lifetime.
This pattern is useful for counters, caches, and implementing closure-like
behavior in C.

.. code-block:: c

    #include <stdio.h>

    void counter(void) {
      static int count = 0;  // initialized once, persists across calls
      int local = 0;         // reinitialized each call

      count++;
      local++;
      printf("static count = %d, local = %d\n", count, local);
    }

    int main(void) {
      for (int i = 0; i < 5; i++) {
        counter();
      }
    }

.. code-block:: console

    $ ./static-closure
    static count = 1, local = 1
    static count = 2, local = 1
    static count = 3, local = 1
    static count = 4, local = 1
    static count = 5, local = 1

Machine Endianness Detection
----------------------------

:Source: `src/c/endian-check <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/endian-check>`_

Endianness determines how multi-byte values are stored in memory. Little-endian
systems store the least significant byte at the lowest address (x86, ARM),
while big-endian systems store the most significant byte first (network byte
order, some RISC processors). Detecting endianness at runtime is essential when
working with binary file formats, network protocols, or cross-platform data
exchange.

.. code-block:: c

    #include <stdio.h>
    #include <stdint.h>

    int is_little_endian(void) {
      uint16_t val = 0x0001;
      return *(uint8_t *)&val == 0x01;
    }

    int main(void) {
      if (is_little_endian()) {
        printf("Little Endian Machine\n");
      } else {
        printf("Big Endian Machine\n");
      }
    }

.. code-block:: console

    $ ./endian-check
    Little Endian Machine

String Splitting
----------------

:Source: `src/c/split-string <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/split-string>`_

Splitting strings by a delimiter is a common operation that C handles through
``strtok()``, which modifies the original string by inserting null terminators.
For reentrant code or when the original string must be preserved, use
``strtok_r()`` or implement custom parsing. This example demonstrates dynamic
memory allocation for storing the resulting tokens.

.. code-block:: c

    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>

    char **split(char *str, char delim) {
      int count = 1;
      for (char *p = str; *p; p++) {
        if (*p == delim) count++;
      }

      char **result = calloc(count + 1, sizeof(char *));
      char delims[2] = {delim, '\0'};
      char *token = strtok(str, delims);
      int i = 0;

      while (token) {
        result[i++] = strdup(token);
        token = strtok(NULL, delims);
      }
      return result;
    }

    int main(void) {
      char str[] = "hello,world,test";
      char **tokens = split(str, ',');

      for (char **p = tokens; *p; p++) {
        printf("%s\n", *p);
        free(*p);
      }
      free(tokens);
    }

.. code-block:: console

    $ ./split-string
    hello
    world
    test

Callback Functions
------------------

:Source: `src/c/callback <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/callback>`_

Callbacks enable flexible, decoupled code by passing function pointers as
arguments. The called function invokes the callback at appropriate points,
allowing the caller to customize behavior without modifying the callee. This
pattern is fundamental to event-driven programming, asynchronous operations,
and plugin architectures. In C, callbacks are implemented using function
pointers with matching signatures.

.. code-block:: c

    #include <stdio.h>

    typedef void (*callback_fn)(int result);

    void on_complete(int result) {
      if (result == 0) {
        printf("Operation succeeded\n");
      } else {
        printf("Operation failed with code %d\n", result);
      }
    }

    void do_work(int should_fail, callback_fn cb) {
      int result = should_fail ? -1 : 0;
      cb(result);
    }

    int main(void) {
      do_work(0, on_complete);  // success
      do_work(1, on_complete);  // failure
    }

.. code-block:: console

    $ ./callback
    Operation succeeded
    Operation failed with code -1

Duff's Device
-------------

:Source: `src/c/duffs-device <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/duffs-device>`_

Duff's device is a loop unrolling technique that exploits C's switch statement
fall-through behavior. By interleaving a switch with a do-while loop, it handles
both the unrolled iterations and the remainder in a single construct. While
modern compilers often perform better automatic optimizations, Duff's device
remains an interesting example of C's flexibility and is occasionally useful
in embedded systems or when compiler optimizations are limited.

.. code-block:: c

    #include <stdio.h>

    void copy_bytes(char *to, const char *from, int count) {
      int n = (count + 7) / 8;
      switch (count % 8) {
        case 0: do { *to++ = *from++;
        case 7:      *to++ = *from++;
        case 6:      *to++ = *from++;
        case 5:      *to++ = *from++;
        case 4:      *to++ = *from++;
        case 3:      *to++ = *from++;
        case 2:      *to++ = *from++;
        case 1:      *to++ = *from++;
                } while (--n > 0);
      }
    }

    int main(void) {
      char src[] = "Hello, Duff!";
      char dst[20] = {0};
      copy_bytes(dst, src, sizeof(src));
      printf("Copied: %s\n", dst);
    }

.. code-block:: console

    $ ./duffs-device
    Copied: Hello, Duff!

Compound Literals
-----------------

Compound literals create unnamed objects with automatic storage duration,
useful for passing temporary arrays or structs to functions without declaring
separate variables. The syntax ``(type){initializer}`` creates an object of
the specified type. Compound literals are lvalues, so their address can be
taken, but they only persist until the end of the enclosing block.

.. code-block:: c

    #include <stdio.h>

    struct point { int x, y; };

    void print_point(struct point *p) {
      printf("(%d, %d)\n", p->x, p->y);
    }

    int main(void) {
      // Pass temporary struct without declaring variable
      print_point(&(struct point){10, 20});

      // Temporary array
      int *arr = (int[]){1, 2, 3, 4, 5};
      for (int i = 0; i < 5; i++) {
        printf("%d ", arr[i]);
      }
      printf("\n");
    }

.. code-block:: console

    $ ./compound-literal
    (10, 20)
    1 2 3 4 5

Designated Initializers
-----------------------

Designated initializers allow initializing specific members of arrays or structs
by name or index, leaving others zero-initialized. This improves code clarity
and maintainability, especially for large structs where only a few fields need
non-default values. Array designators use ``[index] = value`` syntax, while
struct designators use ``.member = value``.

.. code-block:: c

    #include <stdio.h>

    struct config {
      int timeout;
      int retries;
      int verbose;
      char name[32];
    };

    int main(void) {
      // Initialize specific struct members
      struct config cfg = {
        .timeout = 30,
        .name = "default",
        // retries and verbose are zero-initialized
      };

      // Initialize specific array indices
      int arr[10] = {
        [0] = 100,
        [5] = 500,
        [9] = 900,
      };

      printf("timeout=%d, retries=%d\n", cfg.timeout, cfg.retries);
      printf("arr[0]=%d, arr[5]=%d, arr[9]=%d\n", arr[0], arr[5], arr[9]);
    }

.. code-block:: console

    $ ./designated-init
    timeout=30, retries=0
    arr[0]=100, arr[5]=500, arr[9]=900

Flexible Array Members
----------------------

Flexible array members (C99) allow structs to have a variable-length array as
their last member. The struct is allocated with extra space for the array
elements. This pattern is common in network protocols and file formats where
a header contains a count followed by variable data. The flexible array member
has no size in the struct definition and must be allocated dynamically.

.. code-block:: c

    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>

    struct message {
      int length;
      char data[];  // flexible array member
    };

    struct message *create_message(const char *text) {
      int len = strlen(text);
      struct message *msg = malloc(sizeof(struct message) + len + 1);
      msg->length = len;
      strcpy(msg->data, text);
      return msg;
    }

    int main(void) {
      struct message *msg = create_message("Hello, flexible array!");
      printf("Length: %d, Data: %s\n", msg->length, msg->data);
      free(msg);
    }

.. code-block:: console

    $ ./flexible-array
    Length: 22, Data: Hello, flexible array!

Variable-Length Arrays (VLA)
----------------------------

Variable-length arrays (C99) allow array sizes to be determined at runtime.
Unlike dynamically allocated arrays, VLAs have automatic storage duration and
are allocated on the stack. While convenient, VLAs can cause stack overflow
for large sizes and are optional in C11. Use them cautiously for small,
bounded sizes; prefer ``malloc()`` for large or unbounded allocations.

.. code-block:: c

    #include <stdio.h>

    void print_matrix(int rows, int cols, int matrix[rows][cols]) {
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          printf("%3d ", matrix[i][j]);
        }
        printf("\n");
      }
    }

    int main(void) {
      int n = 3;
      int vla[n][n];  // VLA with runtime size

      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          vla[i][j] = i * n + j;
        }
      }

      print_matrix(n, n, vla);
    }

.. code-block:: console

    $ ./vla
      0   1   2
      3   4   5
      6   7   8

Restrict Qualifier
------------------

The ``restrict`` qualifier (C99) tells the compiler that a pointer is the only
way to access the pointed-to object, enabling aggressive optimizations. Without
``restrict``, the compiler must assume pointers might alias (point to overlapping
memory), preventing certain optimizations. Use ``restrict`` when you can
guarantee no aliasing, particularly in performance-critical functions like
``memcpy()``.

.. code-block:: c

    #include <stdio.h>

    // Without restrict, compiler assumes a and b might overlap
    void add_arrays(int *a, int *b, int *result, int n) {
      for (int i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
      }
    }

    // With restrict, compiler can optimize more aggressively
    void add_arrays_fast(int *restrict a, int *restrict b,
                         int *restrict result, int n) {
      for (int i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
      }
    }

    int main(void) {
      int a[] = {1, 2, 3};
      int b[] = {4, 5, 6};
      int result[3];

      add_arrays_fast(a, b, result, 3);
      printf("%d %d %d\n", result[0], result[1], result[2]);
    }

.. code-block:: console

    $ ./restrict
    5 7 9


``_Static_assert`` (C11)
------------------------

:Source: `src/c/static-assert <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/static-assert>`_

Static assertions verify conditions at compile time, catching errors before
runtime. Unlike ``assert()``, ``_Static_assert`` has zero runtime cost and
fails during compilation if the condition is false. Use it to validate type
sizes, struct layouts, and configuration assumptions. C23 allows omitting the
message string.

.. code-block:: c

    #include <stdio.h>
    #include <stdint.h>

    _Static_assert(sizeof(int) >= 4, "int must be at least 32 bits");
    _Static_assert(sizeof(void *) == 8, "64-bit pointers required");

    struct packet {
      uint32_t header;
      uint32_t payload;
    };
    _Static_assert(sizeof(struct packet) == 8, "packet must be 8 bytes");

    int main(void) {
      printf("All static assertions passed\n");
    }

.. code-block:: console

    $ gcc -std=c11 -o static-assert main.c && ./static-assert
    All static assertions passed

``_Generic`` Type Selection (C11)
---------------------------------

:Source: `src/c/generic <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/generic>`_

``_Generic`` enables type-generic macros by selecting expressions based on the
type of the controlling expression. This provides a form of function overloading
in C, commonly used for type-safe math functions and print formatters. The
selection happens at compile time with no runtime overhead.

.. code-block:: c

    #include <stdio.h>
    #include <math.h>

    #define abs_val(x) _Generic((x), \
        int: abs, \
        long: labs, \
        float: fabsf, \
        double: fabs \
    )(x)

    #define type_name(x) _Generic((x), \
        int: "int", \
        float: "float", \
        double: "double", \
        char *: "char *", \
        default: "unknown" \
    )

    int main(void) {
      printf("abs(-5) = %d\n", abs_val(-5));
      printf("abs(-3.14) = %f\n", abs_val(-3.14));

      int i; float f; char *s;
      printf("i is %s, f is %s, s is %s\n",
             type_name(i), type_name(f), type_name(s));
    }

.. code-block:: console

    $ gcc -std=c11 -o generic main.c -lm && ./generic
    abs(-5) = 5
    abs(-3.14) = 3.140000
    i is int, f is float, s is char *

Anonymous Structs and Unions (C11)
----------------------------------

:Source: `src/c/anon-struct <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/anon-struct>`_

Anonymous structs and unions allow nested members to be accessed directly
without an intermediate name. This simplifies code when wrapping related fields
or creating variant types. The inner struct/union members become direct members
of the enclosing struct.

.. code-block:: c

    #include <stdio.h>

    struct vector3 {
      union {
        struct { float x, y, z; };      // anonymous struct
        float components[3];             // array access
      };                                 // anonymous union
    };

    int main(void) {
      struct vector3 v = { .x = 1.0f, .y = 2.0f, .z = 3.0f };

      // Access via named members
      printf("x=%.1f, y=%.1f, z=%.1f\n", v.x, v.y, v.z);

      // Access via array
      for (int i = 0; i < 3; i++) {
        printf("components[%d] = %.1f\n", i, v.components[i]);
      }
    }

.. code-block:: console

    $ gcc -std=c11 -o anon-struct main.c && ./anon-struct
    x=1.0, y=2.0, z=3.0
    components[0] = 1.0
    components[1] = 2.0
    components[2] = 3.0

Atomics (C11)
-------------

:Source: `src/c/atomics <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/atomics>`_

The ``<stdatomic.h>`` header provides atomic types and operations for lock-free
concurrent programming. Atomic operations are indivisible—no other thread can
observe a partial update. Use atomics for simple shared counters and flags;
for complex data structures, prefer mutexes.

.. code-block:: c

    #include <stdio.h>
    #include <stdatomic.h>
    #include <pthread.h>

    atomic_int counter = 0;

    void *increment(void *arg) {
      for (int i = 0; i < 100000; i++) {
        atomic_fetch_add(&counter, 1);
      }
      return NULL;
    }

    int main(void) {
      pthread_t t1, t2;
      pthread_create(&t1, NULL, increment, NULL);
      pthread_create(&t2, NULL, increment, NULL);
      pthread_join(t1, NULL);
      pthread_join(t2, NULL);

      printf("Counter: %d\n", atomic_load(&counter));  // always 200000
    }

.. code-block:: console

    $ gcc -std=c11 -pthread -o atomics main.c && ./atomics
    Counter: 200000

``_Alignas`` and ``_Alignof`` (C11)
-----------------------------------

:Source: `src/c/alignas <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/alignas>`_

``_Alignas`` specifies alignment requirements for variables, useful for SIMD
operations, cache optimization, and hardware interfaces. ``_Alignof`` queries
the alignment requirement of a type. The ``<stdalign.h>`` header provides
``alignas`` and ``alignof`` as convenient aliases.

.. code-block:: c

    #include <stdio.h>
    #include <stdalign.h>
    #include <stdint.h>

    struct aligned_data {
      alignas(64) char cache_line[64];  // aligned to cache line
      alignas(16) float simd_data[4];   // aligned for SIMD
    };

    int main(void) {
      printf("int alignment: %zu\n", alignof(int));
      printf("double alignment: %zu\n", alignof(double));
      printf("cache_line alignment: %zu\n", alignof(struct aligned_data));

      alignas(32) int aligned_int = 42;
      printf("aligned_int address: %p\n", (void *)&aligned_int);
    }

.. code-block:: console

    $ gcc -std=c11 -o alignas main.c && ./alignas
    int alignment: 4
    double alignment: 8
    cache_line alignment: 64
    aligned_int address: 0x7fff5fbff760

``_Noreturn`` Functions (C11)
-----------------------------

:Source: `src/c/noreturn <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/noreturn>`_

The ``_Noreturn`` specifier indicates a function never returns to its caller,
either because it exits the program, loops forever, or always throws. This
enables compiler optimizations and suppresses "missing return" warnings. C23
deprecates ``_Noreturn`` in favor of the ``[[noreturn]]`` attribute.

.. code-block:: c

    #include <stdio.h>
    #include <stdlib.h>
    #include <stdnoreturn.h>

    noreturn void fatal_error(const char *msg) {
      fprintf(stderr, "Fatal: %s\n", msg);
      exit(1);
    }

    int main(void) {
      int *ptr = NULL;
      if (!ptr) {
        fatal_error("null pointer");
      }
      // Compiler knows this is unreachable
      return 0;
    }

.. code-block:: console

    $ gcc -std=c11 -o noreturn main.c && ./noreturn
    Fatal: null pointer

``typeof`` and ``auto`` (C23)
-----------------------------

:Source: `src/c/typeof <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/typeof>`_

C23 adds ``typeof`` to deduce types from expressions and ``auto`` for type
inference in variable declarations. These features reduce repetition and make
code more maintainable, especially with complex types. ``typeof`` was previously
a GNU extension; C23 standardizes it.

.. code-block:: c

    #include <stdio.h>

    #define max(a, b) ({ \
        typeof(a) _a = (a); \
        typeof(b) _b = (b); \
        _a > _b ? _a : _b; \
    })

    int main(void) {
      // auto type inference (C23)
      auto x = 42;           // int
      auto pi = 3.14159;     // double
      auto msg = "hello";    // const char *

      printf("max(3, 7) = %d\n", max(3, 7));
      printf("max(3.5, 2.1) = %f\n", max(3.5, 2.1));

      // typeof for matching types
      int arr[] = {1, 2, 3, 4, 5};
      typeof(arr[0]) sum = 0;
      for (int i = 0; i < 5; i++) {
        sum += arr[i];
      }
      printf("sum = %d\n", sum);
    }

.. code-block:: console

    $ gcc -std=c2x -o typeof main.c && ./typeof
    max(3, 7) = 7
    max(3.5, 2.1) = 3.500000
    sum = 15

``nullptr`` (C23)
-----------------

:Source: `src/c/nullptr-test <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/nullptr-test>`_

C23 introduces ``nullptr`` as a proper null pointer constant with its own type
``nullptr_t``. Unlike ``NULL`` (which is typically ``(void *)0`` or ``0``),
``nullptr`` is unambiguous and type-safe. It cannot be implicitly converted to
integer types, preventing common bugs.

.. code-block:: c

    #include <stdio.h>
    #include <stddef.h>

    void process_int(int n) {
      printf("int: %d\n", n);
    }

    void process_ptr(int *p) {
      printf("ptr: %p\n", (void *)p);
    }

    int main(void) {
      int *p = nullptr;  // clear intent: null pointer

      if (p == nullptr) {
        printf("p is null\n");
      }

      // nullptr prevents accidental integer conversion
      // process_int(nullptr);  // compile error in C23
      process_ptr(nullptr);     // OK
    }

.. code-block:: console

    $ gcc -std=c2x -o nullptr main.c && ./nullptr
    p is null
    ptr: (nil)

``constexpr`` (C23)
-------------------

:Source: `src/c/constexpr-c <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/constexpr-c>`_

C23 extends ``constexpr`` to allow compile-time constant expressions for
variables. Unlike ``const``, which merely prevents modification, ``constexpr``
guarantees the value is computed at compile time and can be used in contexts
requiring constant expressions like array sizes.

.. code-block:: c

    #include <stdio.h>

    constexpr int BUFFER_SIZE = 1024;
    constexpr int DOUBLED = BUFFER_SIZE * 2;

    int main(void) {
      constexpr int local_const = 42;

      // Can use constexpr values for array sizes
      char buffer[BUFFER_SIZE];
      int arr[local_const];

      printf("BUFFER_SIZE = %d\n", BUFFER_SIZE);
      printf("DOUBLED = %d\n", DOUBLED);
      printf("sizeof(buffer) = %zu\n", sizeof(buffer));
    }

.. code-block:: console

    $ gcc -std=c2x -o constexpr main.c && ./constexpr
    BUFFER_SIZE = 1024
    DOUBLED = 2048
    sizeof(buffer) = 1024

Attributes (C23)
----------------

:Source: `src/c/attributes <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/c/attributes>`_

C23 standardizes attributes using the ``[[attribute]]`` syntax, replacing
compiler-specific extensions. Common attributes include ``[[nodiscard]]`` for
return values that shouldn't be ignored, ``[[maybe_unused]]`` to suppress
warnings, and ``[[deprecated]]`` to mark obsolete APIs.

.. code-block:: c

    #include <stdio.h>
    #include <stdlib.h>

    [[nodiscard]] int *allocate(size_t n) {
      return malloc(n * sizeof(int));
    }

    [[deprecated("use new_function instead")]]
    void old_function(void) {
      printf("old function\n");
    }

    void callback([[maybe_unused]] int unused_param) {
      printf("callback called\n");
    }

    int main(void) {
      int *p = allocate(10);  // OK: result used
      // allocate(10);        // warning: nodiscard value ignored

      callback(42);
      free(p);
    }

.. code-block:: console

    $ gcc -std=c2x -o attributes main.c && ./attributes
    callback called

Bit-Precise Integers (C23)
--------------------------

C23 introduces ``_BitInt(N)`` for integers with exactly N bits, useful for
cryptography, hardware interfaces, and memory-constrained systems. Unlike
standard integer types whose sizes vary by platform, bit-precise integers
have guaranteed widths.

.. code-block:: c

    #include <stdio.h>

    int main(void) {
      _BitInt(24) rgb = 0xFF8040;     // exactly 24 bits
      _BitInt(128) big = 0;           // 128-bit integer

      unsigned _BitInt(12) small = 4095;  // 12-bit unsigned

      printf("rgb = 0x%X\n", (unsigned)rgb);
      printf("small max = %u\n", (unsigned)small);
    }

.. code-block:: console

    $ gcc -std=c2x -o bitint main.c && ./bitint
    rgb = 0xFF8040
    small max = 4095
