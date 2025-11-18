==================
C Macro cheatsheet
==================

.. contents:: Table of Contents
    :backlinks: none

Predefined Macros
------------------

============   ================================================
   Macro        descriptions
============   ================================================
  __FILE__      current file name
  __DATE__      current compile date in "MMM DD YYYY"" format.
  __TIME__      current compile time in "HH:MM:SS" format.
  __LINE__      current line number
  __func__      current function name
============   ================================================

.. code-block:: c

    #include <stdio.h>

    int main(int argc, char *argv[])
    {
        int ret = -1;

        printf("__FILE__: %s\n"
               "__DATE__: %s\n"
               "__TIME__: %s\n"
               "__LINE__: %d\n"
               "__func__: %s\n",
               __FILE__, __DATE__, __TIME__, __LINE__, __func__);

        ret = 0;
        return ret;
    }

output:

.. code-block:: bash

    $ cc -g -Wall -o test test.c
    $ ./test
    __FILE__: test.c
    __DATE__: Sep 28 2016
    __TIME__: 10:01:59
    __LINE__: 16
    __func__: main


``DEBUG`` switch
------------------

.. code-block:: c

    #include <stdio.h>

    int main(int argc, char *argv[])
    {
        int ret = -1;

    #ifdef DEBUG
        printf("debug version\n");
    #else
        printf("release version\n");
    #endif

        ret = 0;
        return ret;
    }

output:

.. code-block:: bash

    $ cc -g -Wall -o test test.c
    $ ./test
    release version
    $ cc -g -Wall -DDEBUG -o test test.c
    $ ./test
    debug version


ARRAYSIZE
----------

.. code-block:: c

    #include <stdio.h>

    #define ARRAY_SIZE(a) (sizeof(a) / sizeof(a[0]))

    /*
     * Entry point
     */
    int main(int argc, char *argv[])
    {
        int ret = -1;
        char *pszArr[] = {"Hello", "World", NULL};

        printf("array size: %lu\n", ARRAY_SIZE(pszArr));
        ret = 0;
        return ret;
    }

output:

.. code-block:: bash

    $ cc -g -Wall -o test test.c
    $ ./test
    array size: 3


FOREACH
--------

.. code-block:: c

    #include <stdio.h>

    #define FOREACH(item, arr) \
        for (item=arr; *item; item++)

    /*
     * Entry point
     */
    int main(int argc, char *argv[])
    {
        int ret = -1;
        char *pszArr[] = {"Hello", "World", NULL};
        char **str = NULL;

        FOREACH (str, pszArr) {
            printf("%s ", *str);
        }
        printf("\n");

        ret = 0;
        return ret;
    }

output:

.. code-block:: bash

    $ cc -g -Wall -o test test.c
    $ ./test
    Hello World


ALLOC_STRUCT
-------------

.. code-block:: c

    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <errno.h>

    #define ALLOC_STRUCT(s)  ((s *) malloc(sizeof(s)))
    #define EXPECT_NOT_NULL(i, ...) \
        if (i == NULL) { __VA_ARGS__ }
    #define EXPECT_ALLOC_SUCCESS(i, fmt, ...) \
        EXPECT_NOT_NULL(i, printf(fmt, ##__VA_ARGS__); goto End;)

    typedef struct _foo {
        int hello;
        int world;
    } foo;

    int main(int argc, char *argv[])
    {
        int ret = -1;
        foo *f  = NULL;
        f = ALLOC_STRUCT(foo);
        EXPECT_ALLOC_SUCCESS(f, "err: %s", strerror(errno));
        printf("alloc foo success\n");
        ret = 0;
    End:
        return ret;
    }

output:

.. code-block:: bash

    $ gcc -g -Wall -o test test.c
    $ ./test
    alloc foo success


lambda
-------

.. code-block:: c

    #define lambda(return_type, ...) \
        __extension__ \
        ({ \
            return_type __fn__ __VA_ARGS__ \
            __fn__; \
        })

    /*
     * Entry point
     */
    int main(int argc, char *argv[])
    {
        int ret = -1;
        int (*max) (int, int) =
            lambda (int, (int x, int y) { return x > y ? x : y; });

        printf("lambda: %d\n", max(2,3));

        ret = 0;
        return ret;
    }

output:

.. code-block:: bash

    $ gcc -g -Wall -o test test.c
    $ ./test
    lambda: 3


EXPECT_*
-----------

.. code-block:: c

    #include <stdio.h>                                                                                                                                   [19/1840]
    #include <string.h>
    #include <errno.h>
    #include <sys/types.h>
    #include <sys/stat.h>
    #include <unistd.h>

    #define EXPECT_TRUE(i, ...) \
        if (i != 1) { __VA_ARGS__ }

    #define EXPECT_FALSE(i, ...) \
        if (i != 0) { __VA_ARGS__ }

    #define EXPECT_EQ(i, e, ...) \
        if (i != e) { __VA_ARGS__ }

    #define EXPECT_NEQ(i, e, ...) \
        if (i == e) { __VA_ARGS__ }

    #define EXPECT_LT(i, e, ...) \
        if (i >= e) { __VA_ARGS__ }

    #define EXPECT_LE(i, e, ...) \
        if (i > e) { __VA_ARGS__ }

    #define EXPECT_GT(i, e, ...) \
        if (i <= e) { __VA_ARGS__ }

    #define EXPECT_GE(i, e, ...) \
        if (i < e) { __VA_ARGS__ }

    #define EXPECT_SUCCESS(ret, fmt, ...) \
        EXPECT_GT(ret, 0, \
            printf(fmt, ##__VA_ARGS__); \
            goto End; \
        )

    /*
     * Entry point
     */
    int main(int argc, char *argv[])
    {
        int ret = -1;

        EXPECT_TRUE(1);
        EXPECT_FALSE(0);
        EXPECT_LT(1, 0, printf("check less then fail\n"););
        EXPECT_GT(0, 1, printf("check great then fail\n"););
        EXPECT_SUCCESS(ret, "ret = %d\n", ret);
        ret = 0;
    End:
        return ret;
    }

output:

.. code-block:: bash

    $ cc -g -Wall -o checkerr checkerr.c
    $ ./checkerr
    check less then fail
    check great then fail
    ret = -1


Get struct member `GET_FIELD_PTR`
----------------------------------

.. code-block:: c

    #include <stdio.h>

    #define _GET_FIELD_OFFSET(s, field ) \
        ((short)(long)(&((s *)NULL)->field))

    #define _GET_FIELD_PTR(ps, offset) \
        ((void *)(((char *)ps) + (offset)))

    #define GET_FIELD_PTR(s, ps, field) \
        _GET_FIELD_PTR(ps, _GET_FIELD_OFFSET(s, field))

    typedef struct _foo {
        char name[16];
        int age;
        int gender;
    } foo;

    /*
     * Entry point
     */
    int main(int argc, char *argv[])
    {
        int ret = -1;
        char *name = NULL;
        int *age = NULL, *gender = NULL;
        foo f = {.name="c", .age=44, .gender=0};

        name   = GET_FIELD_PTR(foo, &f, name);
        age    = GET_FIELD_PTR(foo, &f, age);
        gender = GET_FIELD_PTR(foo, &f, gender);

        printf("name: %s\n"
               "age: %d\n"
               "gender: %d\n", name, *age, *gender);

        ret = 0;
        return ret;
    }

output:

.. code-block:: bash

    $ cc -g -Wall -o test test.c
    $ ./test
    name: c
    age: 44
    gender: 0


define ``__attribute__ ((*))``
--------------------------------

.. code-block:: c

    #if __GNUC__ >= 3
    #undef inline
    #define inline         inline __attribute__ ((always_inline))
    #define __noinline      __attribute__ ((noinline))
    #define __pure          __attribute__ ((pure))
    #define __const         __attribute__ ((const))
    #define __noreturn      __attribute__ ((noreturn))
    #define __malloc        __attribute__ ((malloc))
    #define __must_check    __attribute__ ((warn_unused_result))
    #define __deprecated    __attribute__ ((deprecated))
    #define __used          __attribute__ ((used))
    #define __unused        __attribute__ ((unused))
    #define __packed        __attribute__ ((packed))
    #define __align(x)      __attribute__ ((aligned, (x)))
    #define __align_max     __attribute__ ((aligned))
    #define likely(x)       __builtin_expect (!!(x), 1)
    #define unlikely(x)     __builtin_expect (!!(x), 0)
    #else
    #undef inline
    #define __noinline   /* no noinline           */
    #define __pure       /* no pure               */
    #define __const      /* no const              */
    #define __noreturn   /* no noreturn           */
    #define __malloc     /* no malloc             */
    #define __must_check /* no warn_unused_result */
    #define __deprecated /* no deprecated         */
    #define __used       /* no used               */
    #define __unused     /* no unused             */
    #define __packed     /* no packed             */
    #define __align(x)   /* no aligned            */
    #define __align_max  /* no align_max          */
    #define likely(x)    (x)
    #define unlikely(x)  (x)
    #endif
