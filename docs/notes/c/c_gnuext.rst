============================
GNU C Extensions cheatsheet
============================

.. contents:: Table of Contents
    :backlinks: none

with ``__extension__``
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c

    #ifndef __GNUC__
    #error "__GNUC__ not defined"
    #else

    #include <stdio.h>

    /* with __extension__ */
    #define lambda(ret_type, ...)               \
            __extension__                       \
            ({                                  \
                    ret_type __fn__ __VA_ARGS__ \
                    __fn__;                     \
            })

    int main(int argc, char *argv[])
    {
            int a = 5566, b = 9527;
            int c = __extension__ 0b101010;
            int (*max) (int, int) = lambda(int, (int x, int y) {return x > y ? x : y; });

            printf("max(%d, %d) = %d\n", a, b, max(a, b));
            printf("binary const c = %x\n", c);

            return 0;
    }
    #endif

output:

.. code-block:: bash

    $ gcc -g -Wall -std=c99 -pedantic test.c
    $ ./a.out
    max(5566, 9527) = 9527
    binary const c = 2a


without ``__extension__``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c

    #ifndef __GNUC__
    #error "__GNUC__ not defined"
    #else

    #include <stdio.h>

    /* with __extension__ */
    #define lambda(ret_type, ...)               \
            ({                                  \
                    ret_type __fn__ __VA_ARGS__ \
                    __fn__;                     \
            })

    int main(int argc, char *argv[])
    {
            int a = 5566, b = 9527;
            int c = 0b101010;
            int (*max) (int, int) = lambda(int, (int x, int y) {return x > y ? x : y; });

            printf("max(%d, %d) = %d\n", a, b, max(a, b));
            printf("binary const c = %x\n", c);

            return 0;
    }
    #endif

output:

.. code-block:: bash

    $ gcc -g -Wall -pedantic test.c
    test.c: In function 'main':
    test.c:17:17: warning: binary constants are a GCC extension [enabled by default]
             int c = 0b101010;
                     ^
    test.c:18:40: warning: ISO C forbids nested functions [-Wpedantic]
             int (*max) (int, int) = lambda(int, (int x, int y) {return x > y ? x : y; });
                                            ^
    test.c:10:17: note: in definition of macro 'lambda'
                     ret_type __fn__ __VA_ARGS__ \
                     ^
    test.c:9:9: warning: ISO C forbids braced-groups within expressions [-Wpedantic]
             ({                                  \
             ^
    test.c:18:33: note: in expansion of macro 'lambda'
             int (*max) (int, int) = lambda(int, (int x, int y) {return x > y ? x : y; });
                                     ^
    $ ./a.out
    max(5566, 9527) = 9527
    binary const c = 2a


Binary Constants
-----------------

ref: `Binary Constants <https://gcc.gnu.org/onlinedocs/gcc/Binary-constants.html#Binary-constants>`_

.. code-block:: c

    #ifndef __GNUC__
    #error "__GNUC__ not defined"
    #else

    #include <stdio.h>

    int main(int argc, char *argv[])
    {
            int a = 0b0101;
            int b = 0x003a;

            printf("%x, %x\n", a, b);

            return 0;
    }
    #endif

output:

.. code-block:: bash

    $ gcc -g -Wall -pedantic test.c
    test.c: In function 'main':
    test.c:9:17: warning: binary constants are a GCC extension [enabled by default]
             int a = 0b0101;
                     ^
    $ ./a.out
    ./a.out
    5, 3a


Statements and Declarations in Expressions
--------------------------------------------

ref: `Statements and Declarations in Expressions <https://gcc.gnu.org/onlinedocs/gcc-4.1.2/gcc/Statement-Exprs.html#Statement-Exprs>`_

.. code-block:: c

    #ifndef __GNUC__
    #error "__GNUC__ not defined"
    #else

    #include <stdio.h>

    #define square(x)   \
      ({                \
            int y = 0;  \
            y = x * x;  \
            y;          \
       })

    #define max(a, b)           \
      ({                        \
            typeof (a) _a = a;  \
            typeof (b) _b = b;  \
            _a > _b ? _a : _b;  \
       })

    int main(int argc, char *argv[])
    {
            int x = 3;
            int a = 55, b = 66;
            printf("square val: %d\n", square(x));
            printf("max(%d, %d) = %d\n", a, b, max(a, b));
            return 0;
    }

    #endif

output:

.. code-block:: bash

    $ ./a.out
    square val: 9
    max(55, 66) = 66


Locally Declared Labels
------------------------

ref: `Locally Declared Labels <https://gcc.gnu.org/onlinedocs/gcc-4.1.2/gcc/Local-Labels.html#Local-Labels>`_

.. code-block:: c

    #ifndef __GNUC__
    #error "__GNUC__ not defined"
    #else

    #include <stdio.h>

    #define ARRAYSIZE(arr)                          \
      ({                                            \
            size_t size = 0;                        \
            size = sizeof(arr) / sizeof(arr[0]);    \
            size;                                   \
       })

    #define SEARCH(arr, size, target)           \
      ({                                        \
            __label__ found;                    \
            int i = 0;                          \
            int value = -1;                     \
            for (i = 0; i < size; i++) {        \
                    if (arr[i] == target) {     \
                            value = i;          \
                            goto found;         \
                    }                           \
            }                                   \
            value = -1;                         \
            found:                              \
            value;                              \
       })

    int main(int argc, char *argv[])
    {
            int arr[5] = {1, 2, 3, 9527, 5566};
            int target = 9527;

            printf("arr[%d] = %d\n",
                    SEARCH(arr, ARRAYSIZE(arr), target), target);
            return 0;
    }

    #endif

output:

.. code-block:: bash

    $ ./a.out
    arr[3] = 9527


Nested Functions
-----------------

ref: `Nested Functions <https://gcc.gnu.org/onlinedocs/gcc-4.1.2/gcc/Nested-Functions.html#Nested-Functions>`_

.. code-block:: c

    #ifndef __GNUC__
    #error "__GNUC__ not defined"
    #else

    #include <stdio.h>

    int main(int argc, char *argv[])
    {
            double a = 3.0;
            double square(double x) { return x * x; }

            printf("square(%.2lf) = %.2lf\n", a, square(a));
            return 0;
    }
    #endif

output:

.. code-block:: bash

    $ ./a.out
    square(3.00) = 9.00

.. note::

    The nested function can access all the variables of the containing
    function that are visible at the point of its definition. This is
    called **lexical scoping**.

.. code-block:: c

    #ifndef __GNUC__
    #error "__GNUC__ not defined"
    #else

    #include <stdio.h>

    int main(int argc, char *argv[])
    {
            int i = 0;

            void up(void) { i++; }
            printf("i = %d\n", i);
            up();
            printf("i = %d\n", i);
            up();
            printf("i = %d\n", i);

            return 0;
    }
    #endif

output:

.. code-block:: bash

    ./a.out
    i = 0
    i = 1
    i = 2


.. note::

    It is possible to call the nested function from outside the scope of
    its name by storing its address or passing the address to another
    function.


.. code-block:: c

    #ifndef __GNUC__
    #error "__GNUC__ not defined"
    #else

    #include <stdio.h>

    #define ARRAY_SIZE(arr) sizeof(arr) / sizeof(arr[0])
    void print_str(char **arr, int i, char *(*access)(char **arr, int idx))
    {
            char *ptr = NULL;

            if (arr == NULL) return;

            ptr = access(arr, i);
            if (ptr != NULL) {
                    printf("str = %s\n", ptr);
            }
    }

    int main(int argc, char *argv[])
    {
            char *arr[5] = {"Hello", "World", "Foo", "Bar", NULL};
            char *ptr = NULL;
            int i = 0;
            int offset = 1;

            char *access(char **arr, int idx)
            {
                    return arr[idx + offset];
            }

            for (i = 0; i < (ARRAY_SIZE(arr) - offset); i++) {
                    print_str(arr, i, access);
            }

        return 0;
    }
    #endif

output:

.. code-block:: bash

    $ ./a.out
    str = World
    str = Foo
    str = Bar


.. note::

    A nested function can jump to a label inherited from
    a containing function, provided the label is explicitly
    declared in the containing function.

.. code-block:: c

    #ifndef __GNUC__
    #error "__GNUC__ not defined"
    #else

    #include <stdio.h>

    int main(int argc, char *argv[])
    {
            __label__ end;
            int ret = -1, i = 0;

            void up(void)
            {
                    i++;
                    if (i > 2) goto end;
            }
            printf("i = %d\n", i); /* i = 0 */
            up();
            printf("i = %d\n", i); /* i = 1 */
            up();
            printf("i = %d\n", i); /* i = 2 */
            up();
            printf("i = %d\n", i); /* i = 3 */
            up();
            printf("i = %d\n", i); /* i = 4 */
            up();
            ret = 0;
    end:
            return ret;
    }
    #endif

output:

.. code-block:: bash

    $ ./a.out
    i = 0
    i = 1
    i = 2


.. note::

    If you need to declare the nested function before its
    definition, use ``auto`` (which is otherwise meaningless
    for function declarations).

.. code-block:: c

    #ifndef __GNUC__
    #error "__GNUC__ not defined"
    #else

    #include <stdio.h>

    int main(int argc, char *argv[])
    {
            int i = 0;
            auto void up(void);

            void up(void) { i++; }
            printf("i = %d\n", i); /* i = 0 */
            up();
            printf("i = %d\n", i); /* i = 1 */
            up();
            printf("i = %d\n", i); /* i = 2 */
            up();
            return 0;
    }
    #endif

output:

.. code-block:: bash

    $ ./a.out
    i = 0
    i = 1
    i = 2

Referring to a Type with ``typeof``
-------------------------------------

ref: `Referring to a Type with typeof <https://gcc.gnu.org/onlinedocs/gcc-4.1.2/gcc/Typeof.html#Typeof>`_


.. code-block:: c

    #ifndef __GNUC__
    #error "__GNUC__ not defined"
    #else

    #include <stdio.h>

    #define pointer(T)  typeof(T *)
    #define array(T, N) typeof(T [N])

    int g_arr[5];

    int main(int argc, char *argv[])
    {
            int i = 0;
            char **ptr = NULL;

            /* This declares _val with the type of what ptr points to. */
            typeof (*g_arr) val = 5566;
            /* This declares _arr as an array of such values. */
            typeof (*g_arr) arr[3] = {1, 2,3};
            /* This declares y as an array of pointers to characters.*/
            array (pointer (char), 4) str_arr = {"foo", "bar", NULL};

            printf("val: %d\n", val);
            for (i = 0; i < 3; i++) {
                    printf("arr[%d] = %d\n", i, arr[i]);
            }
            for (i = 0, ptr = str_arr; *ptr != NULL ; i++, ptr++) {
                    printf("str_arr[%d] = %s\n", i, *ptr);
            }

            return 0;
    }
    #endif

output:

.. code-block:: bash

    $ ./a.out
    val: 5566
    arr[0] = 1
    arr[1] = 2
    arr[2] = 3
    str_arr[0] = foo
    str_arr[1] = bar


Conditionals with Omitted Operands
-----------------------------------

ref: `Conditionals with Omitted Operands <https://gcc.gnu.org/onlinedocs/gcc-4.1.2/gcc/Conditionals.html#Conditionals>`_

.. note::

    The middle operand in a conditional expression may be
    omitted. Then if the first operand is nonzero, its value
    is the value of the conditional expression.

.. code-block:: c

    #ifndef __GNUC__
    #error "__GNUC__ not defined"
    #else

    #include <stdio.h>

    int main(int argc, char *argv[])
    {
            int x = 1, y = 0;
            int z = -1;

            /* equivalent to x ? x : y */
            z = x ? : y;
            printf("z = %d\n", z);
            return 0;
    }

output:

.. code-block:: bash

    $ ./a.out
    z = 1


Arrays of Length Zero
----------------------

    ref: `Zero-length arrays <https://gcc.gnu.org/onlinedocs/gcc-4.1.2/gcc/Zero-Length.html#Zero-Length>`_

.. note::

    Zero-length arrays are allowed in GNU C. They are very useful as the **last
    element** of a structure which is really a header for a **variable-length**
    object

.. code-block:: c

    #include <stdlib.h>
    #include <errno.h>
    #include <string.h>

    #define CHECK_NULL(ptr, fmt, ...)                   \
            do {                                        \
                    if (!ptr) {                         \
                            printf(fmt, ##__VA_ARGS__); \
                            goto End;                   \
                    }                                   \
            } while(0)

    /* array item has zero length */
    typedef struct _list {
            int len;
            char *item[0];
    } list;

    int main(int argc, char *argv[])
    {

            int ret = -1, len = 3;
            list *p_list = NULL;

            p_list = (list *)malloc(sizeof(list) + sizeof(char *) * len);
            CHECK_NULL(p_list, "malloc fail. [%s]", strerror(errno));

            p_list->item[0] = "Foo";
            p_list->item[1] = "Bar";
            p_list->item[2] = NULL;

            printf("item[0] = %s\n", p_list->item[0]);
            printf("item[1] = %s\n", p_list->item[1]);
            printf("item[2] = %s\n", p_list->item[2]);

            ret = 0;
    End:

            if (p_list)
                    free(p_list);

            return ret;
    }

    #endif

output:

.. code-block:: bash

    $ ./a.out
    item[0] = Foo
    item[1] = Bar
    item[2] = (null)


.. note::

    GCC allows static initialization of flexible array members

.. code-block:: c

    #ifndef __GNUC__
    #error "__GNUC__ not defined"
    #else

    #include <stdio.h>

    typedef struct _list {
            int len;
            int item[];
    } list;

    #define PRINT_LIST(l)                             \
            do {                                      \
                    int i = 0;                        \
                    for (i = 0; i < l.len; i++) {     \
                            printf("%d ", l.item[i]); \
                    }                                 \
                    printf("\n");                     \
            } while(0)

    int main(int argc, char *argv[])
    {
            static list l1 = {3, {1, 2, 3}};
            static list l2 = {5, {1, 2, 3, 4, 5}};

            PRINT_LIST(l1);
            PRINT_LIST(l2);
            return 0;
    }

    #endif

output:

.. code-block:: bash

    $ ./a.out
    1 2 3
    1 2 3 4 5


Variadic Macros
----------------

ref: `Variadic Macros <https://gcc.gnu.org/onlinedocs/gcc-4.1.2/gcc/Variadic-Macros.html#Variadic-Macros>`_

.. code-block:: c

    #ifndef __GNUC__
    #error "__GNUC__ not defined"
    #else

    #include <stdio.h>

    #define DEBUG_C99(fmt, ...)     fprintf(stderr, fmt, ##__VA_ARGS__)
    #define DEBUG_GNU(fmt, args...) fprintf(stderr, fmt, ##args)

    int main(int argc, char *argv[])
    {
            DEBUG_C99("ISO C supported variadic macros\n");
            DEBUG_GNU("GNU C supported variadic macors\n");

            DEBUG_C99("ISO C format str = %s\n", "Foo");
            DEBUG_GNU("GNU C format str = %s\n", "Bar");

            return 0;
    }
    #endif

output:

.. code-block:: bash

    $ ./a.out
    ISO C supported variadic macros
    GNU C supported variadic macors
    ISO C format str = Foo
    GNU C format str = Bar


Compound Literals (cast constructors)
--------------------------------------

ref: `Compound Literals <https://gcc.gnu.org/onlinedocs/gcc-4.9.2/gcc/Compound-Literals.html#Compound-Literals>`_

.. note::

    A compound literal looks like a cast containing an initializer.
    Its value is an object of the type specified in the cast, containing
    the elements specified in the initializer

.. code-block:: c

    #ifndef __GNUC__
    #error "__GNUC__ not defined"
    #else

    #include <stdio.h>

    int main(int argc, char *argv[])
    {
            struct foo {int a; char b[3]; } structure = {};

            /* compound literals (cast constructors )*/

            structure = ((struct foo) { 5566, 'a', 'b'});
            printf("a = %d, b = %s\n", structure.a, structure.b);

            /* equal to */

            struct foo temp = {5566, 'a', 'b'};
            structure = temp;

            printf("a = %d, b = %s\n", structure.a, structure.b);

            return 0;
    }
    #endif

output:

.. code-block:: bash

    $ ./a.out
    a = 5566, b = ab
    a = 5566, b = ab

.. note::

    If the object being initialized has array type of unknown size,
    the size is determined by compound literal size

.. code-block:: c

    #ifndef __GNUC__
    #error "__GNUC__ not defined"
    #else

    #include <stdio.h>

    int main(int argc, char *argv[])
    {
            /* The size is determined by compound literal size */

            static int x[] = (int []) {1, 2, 3, 4, 5};
            static int y[] = (int [3]) {1};
            int i = 0;

            for (i = 0; i < 5; i++) printf("%d ", x[i]);
            printf("\n");

            for (i = 0; i < 3; i++) printf("%d ", y[i]);
            printf("\n");

            /* equal to */

            static int xx[] = {1, 2, 3, 4, 5};
            static int yy[] = {1, 0, 0};

            for (i = 0; i < 5; i++) printf("%d ", xx[i]);
            printf("\n");

            for (i = 0; i < 3; i++) printf("%d ", yy[i]);
            printf("\n");

            return 0;
    }
    #endif

output:

.. code-block:: bash

    ./a.out
    1 2 3 4 5
    1 0 0
    1 2 3 4 5
    1 0 0


Case Ranges
------------

ref: `Case Ranges <https://gcc.gnu.org/onlinedocs/gcc/Case-Ranges.html#Case-Ranges>`_

.. code-block:: c

    #ifndef __GNUC__
    #error "__GNUC__ not defined"
    #else

    #include <stdio.h>

    int foo(int a)
    {
            switch (a) {
                    case 1 ... 3:
                            return 5566;
                    case 4 ... 6:
                            return 9527;
            }
            return 7788;
    }

    int main(int argc, char *argv[])
    {
            int b = 0;

            b = foo(1);
            printf("b = %d\n", b);

            b = foo(5);
            printf("b = %d\n", b);

            b = foo(10);
            printf("b = %d\n", b);

            return 0;
    }
    #endif

output:

.. code-block:: bash

    $ ./a.out
    b = 5566
    b = 9527
    b = 7788

.. warning::

    Be careful, write spaces around the ``...`` (ex: ``r1 ... r2``),
    for otherwise it may be parsed wrong when you use it with integer
    values


Designated Initializers
------------------------

ref: `Initializers <https://gcc.gnu.org/onlinedocs/gcc/Designated-Inits.html#Designated-Inits>`_

Array initializer
~~~~~~~~~~~~~~~~~~

.. code-block:: c

    #ifndef __GNUC__
    #error "__GNUC__ not defined"
    #else

    #include <stdio.h>

    #define ARRLEN 6

    int main(int argc, char *argv[])
    {
            /* ISO C99 support giving the elements in any order */
            int a[ARRLEN] = {[5] = 5566, [2] = 9527};
            /* equal to (ISO C90)*/
            int b[ARRLEN] = {0, 0, 9527, 0, 0, 5566};
            register int i = 0;

            for (i = 0; i < ARRLEN; i++) printf("%d ", a[i]);
            printf("\n");

            for (i = 0; i < ARRLEN; i++) printf("%d ", a[i]);
            printf("\n");

            return 0;
    }
    #endif

output:

.. code-block:: bash

    $ # compile in C90 mode
    $ gcc -std=c90 -pedantic test.c
    test.c: In function 'main':
    test.c:12:26: warning: ISO C90 forbids specifying subobject to initialize [-Wpedantic]
             int a[ARRLEN] = {[5] = 5566, [2] = 9527};
                              ^
    test.c:12:38: warning: ISO C90 forbids specifying subobject to initialize [-Wpedantic]
             int a[ARRLEN] = {[5] = 5566, [2] = 9527};
                                          ^

    $ # compile in C99 mode
    $ gcc -std=c99 -pedantic test.c
    $ ./a.out
    0 0 9527 0 0 5566
    0 0 9527 0 0 5566

.. note::

    GNU C also support to initialize a range of elements to the same value

.. code-block:: c

    #ifndef __GNUC__
    #error "__GNUC__ not defined"
    #else

    #include <stdio.h>

    #define ARRLEN 10

    int main(int argc, char *argv[])
    {
            int arr[ARRLEN] = { [2 ... 5] = 5566, [7 ... 9] = 9527};
            register i = 0;

            for (i = 0; i< ARRLEN; i++) printf("%d ", arr[i]);
            printf("\n");

            return 0;
    }
    #endif

output:

.. code-block:: bash

    $ gcc -pedantic test.c
    test.c: In function 'main':
    test.c:11:32: warning: ISO C forbids specifying range of elements to initialize [-Wpedantic]
             int arr[ARRLEN] = { [2 ... 5] = 5566, [7 ... 9] = 9527};
                                    ^
    test.c:11:29: warning: ISO C90 forbids specifying subobject to initialize [-Wpedantic]
             int arr[ARRLEN] = { [2 ... 5] = 5566, [7 ... 9] = 9527};
                                 ^
    test.c:11:50: warning: ISO C forbids specifying range of elements to initialize [-Wpedantic]
             int arr[ARRLEN] = { [2 ... 5] = 5566, [7 ... 9] = 9527};
                                                      ^
    test.c:11:47: warning: ISO C90 forbids specifying subobject to initialize [-Wpedantic]
             int arr[ARRLEN] = { [2 ... 5] = 5566, [7 ... 9] = 9527};
                                                   ^
    $ ./a.out
    0 0 5566 5566 5566 5566 0 9527 9527 9527

structure & union initializer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c

    #ifndef __GNUC__
    #error "__GNUC__ not defined"
    #else

    #include <stdio.h>

    typedef struct _point {int x, y; } point;
    typedef union _foo {int i; double d; } foo;


    int main(int argc, char *argv[])
    {
            point a = { 5566, 9527 };
            /* GNU C support initialize with .fieldname = */
            point b = { .x = 5566, .y = 9527 };
            /* obsolete since GCC 2.5 */
            point c = { x: 5566, y: 9527 };
            /* specify which element of the union should be used */
            foo bar = { .d = 5566 };

            printf("a.x = %d, a.y = %d\n", a.x, a.y);
            printf("b.x = %d, b.y = %d\n", b.x, b.y);
            printf("c.x = %d, c.y = %d\n", c.x, c.y);
            printf("bar.d = %.2lf\n", bar.d);

            return 0;
    }
    #endif

output:

.. code-block:: bash

    $ gcc -pedantic test.c
    test.c: In function 'main':
    test.c:15:21: warning: ISO C90 forbids specifying subobject to initialize [-Wpedantic]
             point b = { .x = 5566, .y = 9527 };
                         ^
    test.c:15:32: warning: ISO C90 forbids specifying subobject to initialize [-Wpedantic]
             point b = { .x = 5566, .y = 9527 };
                                    ^
    test.c:17:22: warning: obsolete use of designated initializer with ':' [-Wpedantic]
             point c = { x: 5566, y: 9527 };
                          ^
    test.c:17:31: warning: obsolete use of designated initializer with ':' [-Wpedantic]
             point c = { x: 5566, y: 9527 };
                                   ^
    test.c:19:21: warning: ISO C90 forbids specifying subobject to initialize [-Wpedantic]
             foo bar = { .d = 5566 };
                         ^
    test.c:24:9: warning: ISO C90 does not support the '%lf' gnu_printf format [-Wformat=]
             printf("bar.d = %.2lf\n", bar.d);
             ^
    $ a.out
    a.x = 5566, a.y = 9527
    b.x = 5566, b.y = 9527
    c.x = 5566, c.y = 9527
    bar.d = 5566.00


Unnamed Structure and Union Fields
-----------------------------------

.. code-block:: c

    #ifndef __GNUC__
    #error "__GNUC__ not defined"
    #else

    #include <stdio.h>

    struct foo {
            int a;
            union {
                    int b;
                    char byte[4];
            };
            int d;
    };

    int main(int argc, char *argv[])
    {

            struct foo bar = { 0x1a, { 0x2b }, 0x3c };
            int i = 0;

            printf("%x, %x, %x\n", bar.a, bar.b, bar.d);

            /* on little machine, we will get 2b 0 0 0 */
            for (i = 0; i < 4; i++) printf("%x ", bar.byte[i]);
            printf("\n");

            return 0;
    }
    #endif

output:

.. code-block:: console

    # without gcc options -std=c11 will raise warning
    $ gcc -g -Wall -pedantic test.c
    test.c:12:10: warning: ISO C90 doesn't support unnamed structs/unions [-Wpedantic]
             };
              ^
    # with gcc options -std=c11 will not raise warning
    $ gcc -g -Wall -pedantic -std=c11 test.c
    $ ./a.out
    1a, 2b, 3c
    2b 0 0 0

.. note::

    Unnamed field must be a structure or union definition without a tag
    like ``struct { int a; };``. If ``-fms-extensions`` is used, the field
    may also be a definition with a tag such as ``struct foo { int a; };``

.. code-block:: c

    #ifndef __GNUC__
    #error "__GNUC__ not defined"
    #else

    #include <stdio.h>

    struct foo {
            int b;
            int c;
    };

    struct bar {
            int a;
            struct foo;
            int d;
    };

    int main(int argc, char *argv[])
    {
            struct bar baz = { 0x1a, { 0x2b, 0x00 }, 0x3c };

            printf("%x, %x, %x, %x\n", baz.a, baz.b, baz.c, baz.d);

            return 0;
    }
    #endif

output:

.. code-block:: bash

    $ gcc -g -Wall -pedantic -std=c11 -fms-extensions test.c
    $ ./a.out
    1a, 2b, 0, 3c
