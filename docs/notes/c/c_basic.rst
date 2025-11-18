==================
C Basic cheatsheet
==================

.. contents:: Table of Contents
    :backlinks: none

Comma Operator
---------------

.. code-block:: c

    #include <stdio.h>

    #define PRINT(exp...)                           \
            {                                       \
                    exp;                            \
                    printf(#exp " => i = %d\n", i); \
            }

    int main(int argc, char *argv[])
    {
            /* comma just a separators */
            int a = 1, b = 2, c = 3, i = 0;

            printf("(a, b, c) = (%d, %d, %d)\n\n", a, b, c);

            /* comma act as binary operator */
            PRINT( i = (a, b, c) );
            PRINT( i = (a + 5, a + b) );

            /* equivalent to (i = a + 5), a + b */
            PRINT( i = a + 5, a + b );

            return 0;
    }

output:

.. code-block:: bash

    $ ./a.out
    (a, b, c) = (1, 2, 3)

    i = (a, b, c) => i = 3
    i = (a + 5, a + b) => i = 3
    i = a + 5, a + b => i = 6

.. note::

    Comma operator is a **binary operator**, it evaluates its first operand and
    discards the result, and then evaluates the second operand and return this
    value.



Old Style and New Style Function Definition
----------------------------------------------

.. code-block:: c

    #include <stdio.h>

    /* old style function declaration */
    int old_style_add(a, b)
            int a; int b;
    {
            return a + b;
    }

    /* new style function declaration */
    int new_style_add(int a, int b)
    {
            return a + b;
    }

    int main(int argc, char *argv[])
    {

            printf("old_sylte_add = %d\n", old_style_add(5566, 7788));
            printf("new_sylte_add = %d\n", new_style_add(5566, 9527));

            return 0;
    }

output:

.. code-block:: bash

    $ gcc -Wold-style-definition -g -Wall test.c
    test.c: In function 'old_style_add':
    test.c:4:5: warning: old-style function definition [-Wold-style-definition]
     int old_style_add(a, b)
         ^
    $ ./a.out
    old_sylte_add = 13354
    new_sylte_add = 15093


``sizeof(struct {int:-!!(e); })`` Compile Time Assert
-------------------------------------------------------

Reference
~~~~~~~~~~

1. `Stack Overflow <http://stackoverflow.com/q/9229601>`_
2. `/usr/include/linux/kernel.h <https://github.com/torvalds/linux/blob/ff2d8b19a3a62559afba1c53360c8577a7697714/include/linux/kernel.h#L677-L682>`_

.. code-block:: c

    #include <stdio.h>

    #define FORCE_COMPILE_TIME_ERROR_OR_ZERO(e) \
            (sizeof(struct { int:-!!(e); }))

    #define FORCE_COMPILE_TIME_ERROR_OR_NULL(e) \
            ((void *)sizeof(struct { int:-!!(e); }))

    int main(int argc, char *argv[])
    {
            FORCE_COMPILE_TIME_ERROR_OR_ZERO(0);
            FORCE_COMPILE_TIME_ERROR_OR_NULL(NULL);

            return 0;
    }


output:

.. code-block:: bash

    $ gcc test.c
    $ tree .
    .
    |-- a.out
    `-- test.c

    0 directories, 2 files

.. code-block:: c

    #include <stdio.h>

    #define FORCE_COMPILE_TIME_ERROR_OR_ZERO(e) \
            (sizeof(struct { int:-!!(e); }))

    #define FORCE_COMPILE_TIME_ERROR_OR_NULL(e) \
            ((void *)sizeof(struct { int:-!!(e); }))

    int main(int argc, char *argv[])
    {
            int a = 123;

            FORCE_COMPILE_TIME_ERROR_OR_ZERO(a);
            FORCE_COMPILE_TIME_ERROR_OR_NULL(&a);

            return 0;
    }


output:

.. code-block:: bash

    $ gcc test.c
    test.c: In function 'main':
    test.c:4:24: error: bit-field '<anonymous>' width not an integer constant
             (sizeof(struct { int:-!!(e); }))
                            ^
    test.c:13:9: note: in expansion of macro 'FORCE_COMPILE_TIME_ERROR_OR_ZERO'
             FORCE_COMPILE_TIME_ERROR_OR_ZERO(a);
             ^
    test.c:7:32: error: negative width in bit-field '<anonymous>'
             ((void *)sizeof(struct { int:-!!(e); }))
                                    ^
    test.c:14:9: note: in expansion of macro 'FORCE_COMPILE_TIME_ERROR_OR_NULL'
             FORCE_COMPILE_TIME_ERROR_OR_NULL(&a);
             ^


Machine endian check
---------------------

.. code-block:: c

    #include <stdio.h>
    #include <stdint.h>

    static union {
        uint8_t buf[2];
        uint16_t uint16;
    } endian = { {0x00, 0x3a}};

    #define LITTLE_ENDIAN ((char)endian.uint16 == 0x00)
    #define BIG_ENDIAN ((char)endian.uint16 == 0x3a)



    int main(int argc, char *argv[])
    {
        uint8_t buf[2] = {0x00, 0x3a};

        if (LITTLE_ENDIAN) {
            printf("Little Endian Machine: %x\n", ((uint16_t *)buf)[0]);
        } else {
            printf("Big Endian Machine: %x\n", ((uint16_t *)buf)[0]);
        }

        return 0;
    }

output:

.. code-block:: bash

    # on little endian macheine
    $ ${CC} endian_check.c
    $ ./a.out
    Little Endian Machine: 3a00

    # on big endian machine
    $ ${CC} endian_check.c
    $ ./a.out
    Big Endian Machine: 3a


Implement closure via ``static``
--------------------------------

.. code-block:: c

    #include <stdio.h>

    void foo()
    {
        static int s_var = 9527;
        int l_var = 5566;

        l_var++;
        s_var++;
        printf("s_var = %d, l_var = %d\n", s_var, l_var);
    }

    int main(int argc, char *argv[])
    {
        int i = 0;
        for (i=0; i < 5; i++) {
            foo();
        }
        return 0;
    }

output:

.. code-block:: bash

    $ ./a.out
    s_var = 9528, l_var = 5567
    s_var = 9529, l_var = 5567
    s_var = 9530, l_var = 5567
    s_var = 9531, l_var = 5567
    s_var = 9532, l_var = 5567


Split String
------------

.. code-block:: c

    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <assert.h>

    char ** split(char *str, const int sep)
    {
        int num_cut = 1;
        int i = 0;
        char **buf = NULL;
        char *ptr = NULL;
        char delimiters[2] ={sep, '\0'};

        assert(str != NULL);
        printf("pattern = %s\n",str);
        for (ptr = str; *ptr != '\0'; ptr++) {
            if (*ptr == sep){ num_cut++; }
        }
        num_cut++;

        if (NULL == (buf = (char **)calloc(num_cut, sizeof(char *)))) {
            printf("malloc fail\n");
            goto Error;
        }

        ptr = strtok(str, delimiters);
        while (ptr != NULL) {
           buf[i++] = strdup(ptr);
           ptr = strtok(NULL, delimiters);
        }
    Error:
        return buf;
    }

    void free_strlist(char **buf)
    {
        char **ptr = NULL;
        for (ptr = buf; *ptr; ptr++) {
            free(*ptr);
        }
    }

    int main(int argc, char *argv[])
    {
        int ret = -1;
        char *pattern = NULL;
        char **buf = NULL;
        char **ptr = NULL;

        if (argc != 2) {
            printf("Usage: PROG string\n");
            goto Error;
        }

        pattern = argv[1];
        buf = split(pattern, ',');
        for (ptr = buf; *ptr; ptr++) {
            printf("%s\n",*ptr);
        }
        ret = 0;
    Error:
        if (buf) {
            free_strlist(buf);
            buf = NULL;
        }
        return ret;
    }

output:

.. code-block:: console

    $ ./a.out hello,world
    pattern = hello,world
    hello
    world


Callback in C
--------------

.. code-block:: c

    #include <stdio.h>
    #include <string.h>
    #include <errno.h>
    #include <stdint.h>
    #include <sys/types.h>
    #include <sys/stat.h>
    #include <unistd.h>

    #define CHECK_ERR(ret, fmt, ...)        \
        do {                                \
            if (ret < 0) {                  \
                printf(fmt, ##__VA_ARGS__); \
                goto End;                   \
            }                               \
        } while(0)

    void callback(int err)
    {
        if (err < 0) {
            printf("run task fail!\n");
        } else {
            printf("run task success!\n");
        }
    }

    int task(const char *path ,void (*cb)(int err))
    {
        int ret = -1;
        struct stat st = {};

        ret = stat(path, &st);
        CHECK_ERR(ret, "stat(%s) fail. [%s]\n", path, strerror(errno));

        ret = 0;
    End:
        cb(ret); /* run the callback function */
        return ret;
    }


    int main(int argc, char *argv[])
    {
        int ret = -1;
        char *path = NULL;

        if (argc != 2) {
            printf("Usage: PROG [path]\n");
            goto End;
        }
        path = argv[1];
        task(path, callback);
        ret = 0;
    End:
        return ret;
    }

output:

.. code-block:: bash

    $ ${CC} example_callback.c
    $ ./a.out /etc/passwd
    run task success!
    $ ./a.out /etc/passw
    stat(/etc/passw) fail. [No such file or directory]
    run task fail!


Duff's device
--------------

.. code-block:: c

    #include <stdio.h>
    #include <stdlib.h>

    int main(int argc, char* argv[])
    {
        int ret = -1, count = 0;
        int to = 0, from = 0;

        if (argc != 2) {
            printf("Usage: PROG [number]\n");
            goto End;
        }
        count = atoi(argv[1]);
        switch (count % 8) {
            case 0:        do {  to = from++;
            case 7:              to = from++;
            case 6:              to = from++;
            case 5:              to = from++;
            case 4:              to = from++;
            case 3:              to = from++;
            case 2:              to = from++;
            case 1:              to = from++;
                           } while ((count -= 8) > 0);
        }
        printf("get 'to': %d\n", to);
        ret = 0;
    End:
        return ret;
    }

output:

.. code-block:: bash

    $ ./a.out 6
    get 'to': 5
    $ ./a.out
    ./test 19
    get 'to': 18


``switch`` **goto** ``default`` block
---------------------------------------

.. code-block:: c


    #include <stdio.h>

    enum { EVENT_FOO, EVENT_BAR, EVENT_BAZ, EVENT_QUX };

    void demo(int event) {

        switch (event) {
            case EVENT_FOO:
                printf("---> foo event\n");
                break;
            case EVENT_BAR:  while(1) {
                                printf("---> bar event\n");
                                break;
            case EVENT_BAZ:     printf("---> baz event\n");
                                break;
            case EVENT_QUX:     printf("---> qux event\n");
                                break;
                             }
            default:
                printf("default block\n");
        }
    }

    int main(int argc, char *argv[])
    {
        demo(EVENT_FOO); /* will not fall into default block */
        demo(EVENT_BAR); /* will fall into default block */
        demo(EVENT_BAZ); /* will fall into default block */

        return 0;
    }

output:

.. code-block:: bash

    $ ./a.out
    ---> foo event
    ---> bar event
    default block
    ---> baz event
    default block


Simple ``try ... catch`` in C
-------------------------------

.. code-block:: c

    /* cannot distinguish exception */

    #include <stdio.h>
    #include <setjmp.h>

    enum {
        ERR_EPERM = 1,
        ERR_ENOENT,
        ERR_ESRCH,
        ERR_EINTR,
        ERR_EIO
    };

    #define try    do { jmp_buf jmp_env__;     \
                        if (!setjmp(jmp_env__))
    #define catch       else
    #define end    } while(0)

    #define throw(exc) longjmp(jmp_env__, exc)

    int main(int argc, char *argv[])
    {
        int ret = 0;

        try {
            throw(ERR_EPERM);
        } catch {
            printf("get exception!\n");
            ret = -1;
        } end;
        return ret;
    }

output:

.. code-block:: bash

    $ ./a.out
    get exception!


Simple ``try ... catch(exc)`` in C
------------------------------------

.. code-block:: c

    #include <stdio.h>
    #include <string.h>
    #include <setjmp.h>

    enum {
        ERR_EPERM = 1,
        ERR_ENOENT,
        ERR_ESRCH,
        ERR_EINTR,
        ERR_EIO
    };

    #define try    do { jmp_buf jmp_env__;             \
                        switch ( setjmp(jmp_env__) ) { \
                            case 0:
    #define catch(exc)          break;                 \
                            case exc:
    #define end    } } while(0)

    #define throw(exc) longjmp(jmp_env__, exc)

    int main(int argc, char *argv[])
    {
        int ret = 0;

        try {
            throw(ERR_ENOENT);
        } catch(ERR_EPERM) {
            printf("get exception: %s\n", strerror(ERR_EPERM));
            ret = -1;
        } catch(ERR_ENOENT) {
            printf("get exception: %s\n", strerror(ERR_ENOENT));
            ret = -1;
        } catch(ERR_ESRCH) {
            printf("get exception: %s\n", strerror(ERR_ENOENT));
            ret = -1;
        } end;
        return ret;
    }

output:

.. code-block:: bash

    $ ./a.out
    get exception: No such file or directory


Simple ``try ... catch(exc) ... finally`` in C
-----------------------------------------------

.. code-block:: c

    #include <stdio.h>
    #include <string.h>
    #include <setjmp.h>

    enum {
        ERR_EPERM = 1,
        ERR_ENOENT,
        ERR_ESRCH,
        ERR_EINTR,
        ERR_EIO
    };

    #define try  do { jmp_buf jmp_env__  ;             \
                        switch ( setjmp(jmp_env__) ) { \
                            case 0: while(1) {
    #define catch(exc)  	break;                 \
                            case exc:
    #define finally         break; }                   \
                        default:
    #define end  } } while(0)

    #define throw(exc) longjmp(jmp_env__, exc)

    int main(int argc, char *argv[])
    {
        int ret = 0;

        try {
            throw(ERR_ENOENT);
        } catch(ERR_EPERM) {
            printf("get exception: %s\n", strerror(ERR_EPERM));
            ret = -1;
        } catch(ERR_ENOENT) {
            printf("get exception: %s\n", strerror(ERR_ENOENT));
            ret = -1;
        } catch(ERR_ESRCH) {
            printf("get exception: %s\n", strerror(ERR_ENOENT));
            ret = -1;
        } finally {
            printf("finally block\n");
        } end;
        return ret;
    }

output:

.. code-block:: bash

    $ ./a.out
    get exception: No such file or directory
    finally block

ref: `Exceptions in C with Longjmp and Setjmp <http://www.di.unipi.it/~nids/docs/longjump_try_trow_catch.html>`_


Implement a **Task** Chain
---------------------------

.. code-block:: c

    #include <stdio.h>

    typedef enum {
        TASK_FOO = 0,
        TASK_BAR,
        TASK_BAZ,
        TASK_NUM
    } task_set;

    #define NUM_TASKS TASK_NUM
    #define LIST_ADD(list, ptr)       \
        do {                          \
            if (!list) {              \
                (list) = (ptr);       \
                ptr->prev = NULL;     \
                ptr->next = NULL;     \
            } else {                  \
                (list)->prev = ptr;   \
                (ptr)->next = (list); \
                (ptr)->prev = NULL;   \
                (list) = (ptr);       \
            }                         \
        } while(0)

    struct task {
        task_set task_label;
        void (*task) (void);
        struct task *next, *prev;
    };

    static void foo(void) { printf("Foo task\n"); }
    static void bar(void) { printf("Bar task\n"); }
    static void baz(void) { printf("Baz task\n"); }

    struct task task_foo = { TASK_FOO, foo, NULL, NULL };
    struct task task_bar = { TASK_BAR, bar, NULL, NULL };
    struct task task_baz = { TASK_BAZ, baz, NULL, NULL };
    static struct task *task_list = NULL;

    static void register_task(struct task *t)
    {
        LIST_ADD(task_list, t);
    }

    static void lazy_init(void)
    {
        static init_done = 0;

        if (init_done == 0) {
            init_done = 1;

            /* register tasks */
            register_task(&task_foo);
            register_task(&task_bar);
            register_task(&task_baz);
        }
    }

    static void init_tasks(void) {
        lazy_init();
    }

    static struct task * get_task(task_set label)
    {
        struct task *t = task_list;
        while (t) {
            if (t->task_label == label) {
                return t;
            }
            t = t->next;
        }
        return NULL;
    }

    #define RUN_TASK(label, ...)              \
        do {                                  \
            struct task *t = NULL;            \
            t = get_task(label);              \
            if (t) { t-> task(__VA_ARGS__); } \
        } while(0)


    int main(int argc, char *argv[])
    {
        int i = 0;
        init_tasks();

        /* run chain of tasks */
        for (i=0; i<NUM_TASKS; i++) {
            RUN_TASK(i);
        }
        return 0;
    }

output:

.. code-block:: bash

    $ ./a.out
    Foo task
    Bar task
    Baz task
