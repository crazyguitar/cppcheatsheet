==========
C Makefile
==========

.. meta::
   :description: GNU Make reference for C projects covering automatic variables, string functions, pattern rules, shared libraries, and recursive builds.
   :keywords: Makefile, GNU Make, C build system, automatic variables, pattern rules, shared library, static library, recursive make

.. contents:: Table of Contents
    :backlinks: none

Introduction
------------

GNU Make is the standard build automation tool for C and C++ projects, using
Makefiles to define compilation rules and dependencies. Make tracks file
modification times to rebuild only what has changed, dramatically speeding up
incremental builds. This reference covers essential Make features from automatic
variables and string functions to building shared libraries and recursive project
structures. Understanding Make is fundamental for systems programming and
contributes to faster development cycles.

Automatic Variables
-------------------

Automatic variables are set by Make for each rule and provide access to target
and prerequisite names without hardcoding. These variables enable writing generic
pattern rules that work across different files. The most common are ``$@`` for
the target, ``$<`` for the first prerequisite, and ``$^`` for all prerequisites
with duplicates removed.

+------------------------+-----------------------------------------------------------------+
|   automatic variables  |        descriptions                                             |
+------------------------+-----------------------------------------------------------------+
|         $@             | The file name of the target                                     |
+------------------------+-----------------------------------------------------------------+
|         $<             | The name of the first prerequisite                              |
+------------------------+-----------------------------------------------------------------+
|         $^             | The names of all the prerequisites                              |
+------------------------+-----------------------------------------------------------------+
|         $\+            | prerequisites listed more than once are duplicated in the order |
+------------------------+-----------------------------------------------------------------+

Makefile

.. code-block:: make

    .PHONY: all

    all: hello world

    hello world: foo foo foo bar bar
            @echo "== target: $@ =="
            @echo $<
            @echo $^
            @echo $+

    foo:
            @echo "Hello foo"

    bar:
            @echo "Hello Bar"

output

.. code-block:: bash

    Hello foo
    Hello Bar
    == target: hello ==
    foo
    foo bar
    foo foo foo bar bar
    == target: world ==
    foo
    foo bar
    foo foo foo bar bar


using ``$(warning text)`` check make rules (for debug)
--------------------------------------------------------

The ``$(warning text)`` function prints a message during Makefile parsing without
stopping execution, making it invaluable for debugging complex Makefiles. By
placing warnings at strategic points, you can trace variable expansion order,
understand when rules are evaluated, and identify why variables have unexpected
values. Unlike ``$(error)``, warnings allow the build to continue.

.. code-block:: make

    $(warning Top level warning)

    FOO := $(warning FOO variable)foo
    BAR  = $(warning BAR variable)bar

    $(warning target)target: $(warning prerequisite list)Makefile $(BAR)
            $(warning tagrget script)
            @ls
    $(BAR):

output

.. code-block:: bash

    Makefile:1: Top level warning
    Makefile:3: FOO variable
    Makefile:6: target
    Makefile:6: prerequisite list
    Makefile:6: BAR variable
    Makefile:9: BAR variable
    Makefile:7: tagrget script
    Makefile

String Functions
----------------

Make provides powerful string manipulation functions for transforming file lists,
extracting components, and pattern matching. These functions operate on
whitespace-separated words and are essential for deriving object file lists from
sources, filtering files by pattern, and performing substitutions. Common functions
include ``subst``, ``patsubst``, ``filter``, ``filter-out``, and word extraction
functions.

Makefile

.. code-block:: make

    SRC      = hello_foo.c hello_bar.c foo_world.c bar_world.c

    SUBST    = $(subst .c,,$(SRC))

    SRCST    = $(SRC:.c=.o)
    PATSRCST = $(SRC:%.c=%.o)
    PATSUBST = $(patsubst %.c, %.o, $(SRC))

    .PHONY: all

    all: sub filter findstring words word wordlist

    sub:
            @echo "== sub example =="
            @echo "SUBST: " $(SUBST)
            @echo "SRCST: " $(SRCST)
            @echo "PATSRCST: " $(PATSRCST)
            @echo "PATSUBST: " $(PATSUBST)
            @echo ""

    filter:
            @echo "== filter example =="
            @echo "filter: " $(filter hello_%, $(SRC))
            @echo "filter-out: $(filter-out hello_%, $(SRC))"
            @echo ""

    findstring:
            @echo "== findstring example =="
            @echo "Res: " $(findstring hello, hello world)
            @echo "Res: " $(findstring hello, ker)
            @echo "Res: " $(findstring world, worl)
            @echo ""

    words:
            @echo "== words example =="
            @echo "num of words: "$(words $(SRC))
            @echo ""


    word:
            @echo "== word example =="
            @echo "1st word: " $(word 1,$(SRC))
            @echo "2nd word: " $(word 2,$(SRC))
            @echo "3th word: " $(word 3,$(SRC))
            @echo ""


    wordlist:
            @echo "== wordlist example =="
            @echo "[1:3]:"$(wordlist 1,3,$(SRC))
            @echo ""

output

.. code-block:: bash

    $ make
    == sub example ==
    SUBST:  hello_foo hello_bar foo_world bar_world
    SRCST:  hello_foo.o hello_bar.o foo_world.o bar_world.o
    PATSRCST:  hello_foo.o hello_bar.o foo_world.o bar_world.o
    PATSUBST:  hello_foo.o hello_bar.o foo_world.o bar_world.o

    == filter example ==
    filter:  hello_foo.c hello_bar.c
    filter-out: foo_world.c bar_world.c

    == findstring example ==
    Res:  hello
    Res:
    Res:

    == words example ==
    num of words: 4

    == word example ==
    1st word:  hello_foo.c
    2nd word:  hello_bar.c
    3th word:  foo_world.c

    == wordlist example ==
    [1:3]:hello_foo.c hello_bar.c foo_world.c


using ``$(sort list)`` sort list and remove duplicates
-------------------------------------------------------

The ``$(sort list)`` function sorts words lexicographically and removes duplicates
in a single operation. This is particularly useful for normalizing file lists,
ensuring consistent ordering in builds, and eliminating redundant entries that
might cause duplicate symbol errors during linking.

Makefile

.. code-block:: make

    SRC = foo.c bar.c ker.c foo.h bar.h ker.h

    .PHONY: all

    all:
            @echo $(suffix $(SRC))
            @echo $(sort $(suffix $(SRC)))

output

.. code-block:: bash

    $ make
    .c .c .c .h .h .h
    .c .h

Single Dollar Sign and Double Dollar Sign
------------------------------------------

Understanding dollar sign escaping is crucial when mixing Make variables with
shell commands. A single ``$`` references Make variables and is expanded before
the shell sees the command. A double ``$$`` escapes to a single ``$`` for the
shell, allowing access to shell variables and loop counters within recipes. This
distinction is a common source of confusion in Makefiles.

+-------------+-----------------------------------------+
| dollar sign | descriptions                            |
+-------------+-----------------------------------------+
|     ``$``   | reference a make variable using ``$``   |
+-------------+-----------------------------------------+
|    ``$$``   | reference a shell variable using ``$$`` |
+-------------+-----------------------------------------+

Makefile

.. code-block:: make

    LIST = one two three

    .PHONY: all single_dollar double_dollar

    all: single_dollar double_dollar

    double_dollar:
            @echo "=== double dollar sign example ==="
            @for i in $(LIST); do \
                    echo $$i;     \
            done

    single_dollar:
            @echo "=== single dollar sign example ==="
            @for i in $(LIST); do  \
                    echo $i;     \
            done


output

.. code-block:: bash

    $ make
    === single dollar sign example ===



    === double dollar sign example ===
    one
    two
    three


Build Executable Files Respectively
------------------------------------

This pattern demonstrates building multiple executables from individual source
files using implicit rules and wildcard functions. The ``$(wildcard)`` function
finds all matching files, and substitution patterns derive object and executable
names. This approach scales automatically as new source files are added to the
directory.

directory layout

.. code-block:: bash

    .
    |-- Makefile
    |-- bar.c
    |-- bar.h
    |-- foo.c
    `-- foo.h

Makefile

.. code-block:: make

    # CFLAGS: Extra flags to give to the C compiler
    CFLAGS   += -Werror -Wall -O2 -g
    SRC       = $(wildcard *.c)
    OBJ       = $(SRC:.c=.o)
    EXE       = $(subst .c,,$(SRC))

    .PHONY: all clean

    all: $(OBJ) $(EXE)

    clean:
        rm -rf *.o *.so *.a *.la $(EXE)


output

.. code-block:: bash

    $ make
    cc -Werror -Wall -O2 -g   -c -o foo.o foo.c
    cc -Werror -Wall -O2 -g   -c -o bar.o bar.c
    cc   foo.o   -o foo
    cc   bar.o   -o bar


using ``$(eval)`` predefine variables
--------------------------------------

The ``$(eval)`` function parses its argument as Makefile syntax, enabling dynamic
rule and variable generation. Combined with ``$(call)`` and ``$(foreach)``, it
allows creating per-target variables and rules programmatically. Without ``eval``,
the generated text is treated as literal output rather than Makefile constructs,
causing syntax errors.

without ``$(eval)``

.. code-block:: make

    SRC = $(wildcard *.c)
    EXE = $(subst .c,,$(SRC))

    define PROGRAM_template
    $1_SHARED = lib$(strip $1).so
    endef

    .PHONY: all

    $(foreach exe, $(EXE), $(call PROGRAM_template, $(exe)))

    all:
            @echo $(foo_SHARED)
            @echo $(bar_SHARED)

output

.. code-block:: bash

    $ make
    Makefile:11: *** missing separator.  Stop.


with ``$(evall)``

.. code-block:: make

    CFLAGS  += -Wall -g -O2 -I./include
    SRC = $(wildcard *.c)
    EXE = $(subst .c,,$(SRC))

    define PROGRAM_template
    $1_SHARED = lib$(strip $1).so
    endef

    .PHONY: all

    $(foreach exe, $(EXE), $(eval $(call PROGRAM_template, $(exe))))

    all:
            @echo $(foo_SHARED)
            @echo $(bar_SHARED)

output

.. code-block:: bash

    $ make
    libfoo.so
    libbar.so


Build Subdir and Link Together
-------------------------------

This pattern compiles source files from a subdirectory and links them into a
single executable. The ``-I`` flag adds include paths, and pattern rules handle
the compilation. This structure separates headers, sources, and build artifacts
while maintaining a clean project layout.

directory layout

.. code-block:: bash

    .
    |-- Makefile
    |-- include
    |   `-- foo.h
    `-- src
        |-- foo.c
        `-- main.c

Makefile

.. code-block:: make

    CFLAGS  += -Wall -g -O2 -I./include
    SRC     = $(wildcard src/*.c)
    OBJ     = $(SRC:.c=.o)
    EXE     = main

    .PHONY: all clean

    all: $(OBJ) $(EXE)

    $(EXE): $(OBJ)
            $(CC) $(LDFLAGS) -o $@ $^

    %.o: %.c
            $(CC) $(CFLAGS) -c $< -o $@

    clean:
            rm -rf *.o *.so *.a *.la $(EXE) src/*.o src/*.so src/*a

output

.. code-block:: bash

    $ make
    cc -Wall -g -O2 -I./include -c src/foo.c -o src/foo.o
    cc -Wall -g -O2 -I./include -c src/main.c -o src/main.o
    cc  -o main src/foo.o src/main.o


Build Shared Library
---------------------

Shared libraries (``.so`` files) allow code reuse across multiple programs and
reduce memory usage through shared memory mapping. Building them requires
position-independent code (``-fPIC``) and the ``-shared`` linker flag. The
``-Wl,-soname`` option embeds the library's canonical name for runtime linking,
enabling version management through symbolic links.

directory layout

.. code-block:: bash

    .
    |-- Makefile
    |-- include
    |   `-- common.h
    `-- src
        |-- bar.c
        `-- foo.c

Makefile

.. code-block:: make

    SONAME    = libfoobar.so.1
    SHARED    = src/libfoobar.so.1.0.0
    SRC       = $(wildcard src/*.c)
    OBJ       = $(SRC:.c=.o)

    CFLAGS    += -Wall -Werror -fPIC -O2 -g -I./include
    LDFLAGS   += -shared -Wl,-soname,$(SONAME)

    .PHONY: all clean

    all: $(SHARED) $(OBJ)

    $(SHARED): $(OBJ)
            $(CC) $(LDFLAGS) -o $@ $^

    %.o: %.c
            $(CC) $(CFLAGS) -c $^ -o $@

    clean:
            rm -rf src/*.o src/*.so.* src/*.a src/*.la

output

.. code-block:: bash

    $ make
    cc -Wall -Werror -fPIC -O2 -g -I./include -c src/foo.c -o src/foo.o
    cc -Wall -Werror -fPIC -O2 -g -I./include -c src/bar.c -o src/bar.o
    cc -shared -Wl,-soname,libfoobar.so.1 -o src/libfoobar.so.1.0.0 src/foo.o src/bar.o


Build Shared and Static Library
--------------------------------

This example builds both static (``.a``) and shared (``.so``) libraries from the
same source files. Static libraries are archives of object files linked directly
into executables, while shared libraries are loaded at runtime. The symbolic
links (``libfoo.so`` → ``libfoo.so.1`` → ``libfoo.so.1.0.0``) follow the standard
versioning convention for ABI compatibility.

directory layout

.. code-block:: bash

    .
    |-- Makefile
    |-- include
    |   |-- bar.h
    |   `-- foo.h
    `-- src
        |-- Makefile
        |-- bar.c
        `-- foo.c

Makefile

.. code-block:: make

    SUBDIR = src

    .PHONY: all clean $(SUBDIR)

    all: $(SUBDIR)

    clean: $(SUBDIR)

    $(SUBDIR):
            make -C $@ $(MAKECMDGOALS)


src/Makefile

.. code-block:: bash

    SRC      = $(wildcard *.c)
    OBJ      = $(SRC:.c=.o)
    LIB      = libfoobar

    STATIC   = $(LIB).a
    SHARED   = $(LIB).so.1.0.0
    SONAME   = $(LIB).so.1
    SOFILE   = $(LIB).so

    CFLAGS  += -Wall -Werror -g -O2 -fPIC -I../include
    LDFLAGS += -shared -Wl,-soname,$(SONAME)

    .PHONY: all clean

    all: $(STATIC) $(SHARED) $(SONAME) $(SOFILE)

    $(SOFILE): $(SHARED)
            ln -sf $(SHARED) $(SOFILE)

    $(SONAME): $(SHARED)
            ln -sf $(SHARED) $(SONAME)

    $(SHARED): $(STATIC)
            $(CC) $(LDFLAGS) -o $@ $<

    $(STATIC): $(OBJ)
            $(AR) $(ARFLAGS) $@ $^

    %.o: %.c
            $(CC) $(CFLAGS) -c -o $@ $<

    clean:
            rm -rf *.o *.a *.so *.so.*

output

.. code-block:: bash

    $ make
    make -C src
    make[1]: Entering directory '/root/test/src'
    cc -Wall -Werror -g -O2 -fPIC -I../include -c -o foo.o foo.c
    cc -Wall -Werror -g -O2 -fPIC -I../include -c -o bar.o bar.c
    ar rv libfoobar.a foo.o bar.o
    ar: creating libfoobar.a
    a - foo.o
    a - bar.o
    cc -shared -Wl,-soname,libfoobar.so.1 -o libfoobar.so.1.0.0 libfoobar.a
    ln -sf libfoobar.so.1.0.0 libfoobar.so.1
    ln -sf libfoobar.so.1.0.0 libfoobar.so
    make[1]: Leaving directory '/root/test/src'


Build Recursively
------------------

Recursive Make invokes Make in subdirectories, enabling modular project structures
where each component has its own Makefile. The ``$(MAKE)`` variable ensures proper
propagation of flags and job server connections for parallel builds. The
``$(MAKECMDGOALS)`` variable passes the original target (e.g., ``clean``) to
subdirectory Makefiles.

directory layout

.. code-block:: bash

    .
    |-- Makefile
    |-- include
    |   `-- common.h
    |-- src
    |   |-- Makefile
    |   |-- bar.c
    |   `-- foo.c
    `-- test
        |-- Makefile
        `-- test.c

Makefile

.. code-block:: make

    SUBDIR = src test

    .PHONY: all clean $(SUBDIR)

    all: $(SUBDIR)

    clean: $(SUBDIR)

    $(SUBDIR):
            $(MAKE) -C $@ $(MAKECMDGOALS)


src/Makefile

.. code-block:: make

    SONAME   = libfoobar.so.1
    SHARED   = libfoobar.so.1.0.0
    SOFILE   = libfoobar.so

    CFLAGS  += -Wall -g -O2 -Werror -fPIC -I../include
    LDFLAGS += -shared -Wl,-soname,$(SONAME)

    SRC      = $(wildcard *.c)
    OBJ      = $(SRC:.c=.o)

    .PHONY: all clean

    all: $(SHARED) $(OBJ)

    $(SHARED): $(OBJ)
            $(CC) $(LDFLAGS) -o $@ $^
            ln -sf $(SHARED) $(SONAME)
            ln -sf $(SHARED) $(SOFILE)

    %.o: %.c
            $(CC) $(CFLAGS) -c $< -o $@

    clean:
            rm -rf *.o *.so.* *.a *.so

test/Makefile

.. code-block:: make

    CFLAGS    += -Wall -Werror -g -I../include
    LDFLAGS   += -Wall -L../src -lfoobar

    SRC        = $(wildcard *.c)
    OBJ        = $(SRC:.c=.o)
    EXE        = test_main

    .PHONY: all clean

    all: $(OBJ) $(EXE)

    $(EXE): $(OBJ)
            $(CC) -o $@ $^ $(LDFLAGS)

    %.o: %.c
            $(CC) $(CFLAGS) -c $< -o $@

    clean:
            rm -rf *.so *.o *.a $(EXE)

output

.. code-block:: bash

    $ make
    make -C src
    make[1]: Entering directory '/root/proj/src'
    cc -Wall -g -O2 -Werror -fPIC -I../include -c foo.c -o foo.o
    cc -Wall -g -O2 -Werror -fPIC -I../include -c bar.c -o bar.o
    cc -shared -Wl,-soname,libfoobar.so.1 -o libfoobar.so.1.0.0 foo.o bar.o
    ln -sf libfoobar.so.1.0.0 libfoobar.so.1
    ln -sf libfoobar.so.1.0.0 libfoobar.so
    make[1]: Leaving directory '/root/proj/src'
    make -C test
    make[1]: Entering directory '/root/proj/test'
    cc -Wall -Werror -g -I../include -c test.c -o test.o
    cc -o test_main test.o -Wall -L../src -lfoobar
    make[1]: Leaving directory '/root/proj/test'
    $ tree .
    .
    |-- Makefile
    |-- include
    |   `-- common.h
    |-- src
    |   |-- Makefile
    |   |-- bar.c
    |   |-- bar.o
    |   |-- foo.c
    |   |-- foo.o
    |   |-- libfoobar.so -> libfoobar.so.1.0.0
    |   |-- libfoobar.so.1 -> libfoobar.so.1.0.0
    |   `-- libfoobar.so.1.0.0
    `-- test
        |-- Makefile
        |-- test.c
        |-- test.o
        `-- test_main

    3 directories, 14 files


Replace Current Shell
----------------------

Make normally uses ``/bin/sh`` to execute recipe commands, but the ``SHELL``
variable can be overridden to use any interpreter. This enables writing recipes
in Python, Perl, or other languages directly in the Makefile, though it reduces
portability and should be used sparingly.

.. code-block:: make

    OLD_SHELL := $(SHELL)
    SHELL = /usr/bin/python

    .PHONY: all

    all:
            @import os; print os.uname()[0]

output

.. code-block:: bash

    $ make
    Linux


One Line Condition
-------------------

The ``$(if condition,then-part,else-part)`` function provides inline conditional
logic for variable assignments and function calls. The condition is true if it
expands to any non-empty string. This is useful for setting defaults, conditional
flags, and adapting behavior based on variable values without verbose ``ifdef``
blocks.

syntax: ``$(if cond, then part, else part)``

Makefile

.. code-block:: make

    VAR =
    IS_EMPTY = $(if $(VAR), $(info not empty), $(info empty))

    .PHONY: all

    all:
            @echo $(IS_EMPTY)

output

.. code-block:: bash

    $ make
    empty

    $ make VAR=true
    not empty


Using Define to Control CFLAGS
-------------------------------

The ``ifdef`` directive tests whether a variable is defined, enabling conditional
compilation flags. This pattern allows enabling debug symbols, sanitizers, or
optimization levels from the command line without modifying the Makefile. Variables
can be set via ``make VAR=value`` or environment variables, providing flexible
build configuration.

Makefile

.. code-block:: make

    CFLAGS += -Wall -Werror -g -O2
    SRC     = $(wildcard *.c)
    OBJ     = $(SRC:.c=.o)
    EXE     = $(subst .c,,$(SRC))

    ifdef DEBUG
    CFLAGS += -DDEBUG
    endif

    .PHONY: all clean

    all: $(OBJ) $(EXE)

    clean:
            rm -rf $(OBJ) $(EXE)


output

.. code-block:: bash

    $ make
    cc -Wall -Werror -g -O2   -c -o foo.o foo.c
    cc   foo.o   -o foo
    $ make DEBUG=1
    cc -Wall -Werror -g -O2 -DDEBUG   -c -o foo.o foo.c
    cc   foo.o   -o foo
