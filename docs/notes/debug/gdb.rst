===
GDB
===

.. meta::
   :description: GDB debugger tutorial with commands for breakpoints, stepping, stack traces, memory inspection, watchpoints, reverse debugging, and CUDA GPU debugging with cuda-gdb.
   :keywords: GDB tutorial, GDB commands, debugging C++, breakpoints, watchpoints, stack trace, backtrace, memory inspection, reverse debugging, GDB TUI, core dump, segfault debugging, cuda-gdb, CUDA debugging, GPU debugging

.. contents:: Table of Contents
    :backlinks: none

Introduction
------------

GDB (GNU Debugger) is the standard debugger for C and C++ programs on Unix-like
systems. It allows you to pause program execution, inspect variables and memory,
step through code line by line, and understand why programs crash or behave
unexpectedly. Mastering GDB is essential for diagnosing segmentation faults,
memory corruption, race conditions, and logic errors.

Compile your program with ``-g`` to include debug symbols, which map machine
code back to source lines and variable names. Without debug symbols, GDB can
still debug at the assembly level, but source-level debugging is far more
productive.

Command Shortcuts
-----------------

Most GDB commands have single or two-letter shortcuts:

.. code-block:: text

    Command             Shortcut    Description
    ─────────────────────────────────────────────────────────
    run                 r           Run program
    continue            c           Continue execution
    next                n           Step over (next line)
    step                s           Step into function
    finish              fin         Run until function returns
    until               u           Run until line/address
    break               b           Set breakpoint
    delete              d           Delete breakpoint
    info                i           Show info (i b = info breakpoints)
    print               p           Print variable
    backtrace           bt          Show call stack
    frame               f           Select stack frame
    list                l           List source code
    quit                q           Exit GDB
    ─────────────────────────────────────────────────────────
    reverse-next        rn          Reverse step over
    reverse-step        rs          Reverse step into
    reverse-continue    rc          Reverse continue

Press Enter to repeat the last command (useful for repeated ``n`` or ``s``).

Getting Started
---------------

Compile with debug symbols and load into GDB:

.. code-block:: bash

    $ gcc -g -Wall -o myprogram main.c
    $ gdb ./myprogram

Common startup commands:

.. code-block:: bash

    $ gdb ./myprogram              # start GDB with executable
    $ gdb ./myprogram core         # debug a core dump
    $ gdb -p 1234                  # attach to running process
    $ gdb --args ./prog arg1 arg2  # pass arguments to program

Text User Interface
-------------------

GDB's TUI mode provides a visual interface showing source code alongside the
command prompt, similar to an IDE. This makes it easier to see context while
stepping through code.

Key bindings:

- ``Ctrl-x a`` - Toggle TUI mode on/off
- ``Ctrl-x o`` - Switch focus between windows
- ``Ctrl-x 1`` - Single window layout (source + command)
- ``Ctrl-x 2`` - Two window layout (source + assembly + command)
- ``Ctrl-l`` - Refresh display (useful if output corrupts the screen)

.. code-block:: bash

    (gdb) tui enable       # enable TUI mode
    (gdb) layout src       # show source window
    (gdb) layout asm       # show assembly window
    (gdb) layout split     # show source and assembly
    (gdb) layout regs      # show registers

Running Programs
----------------

Control program execution:

.. code-block:: bash

    (gdb) run                    # run program from start
    (gdb) run arg1 arg2          # run with arguments
    (gdb) start                  # run and stop at main()
    (gdb) continue               # continue after breakpoint (c)
    (gdb) next                   # step over function calls (n)
    (gdb) step                   # step into function calls (s)
    (gdb) finish                 # run until current function returns
    (gdb) until 42               # run until line 42
    (gdb) kill                   # stop the program

Breakpoints
-----------

Breakpoints pause execution at specific locations. GDB supports line breakpoints,
function breakpoints, and conditional breakpoints that only trigger when an
expression is true.

.. code-block:: bash

    (gdb) break main             # break at function main
    (gdb) break 42               # break at line 42 in current file
    (gdb) break file.c:42        # break at line 42 in file.c
    (gdb) break func if x > 10   # conditional breakpoint
    (gdb) tbreak main            # temporary breakpoint (auto-deleted)

    (gdb) info breakpoints       # list all breakpoints
    (gdb) delete 1               # delete breakpoint 1
    (gdb) delete                 # delete all breakpoints
    (gdb) disable 1              # disable breakpoint 1
    (gdb) enable 1               # enable breakpoint 1
    (gdb) clear func             # clear breakpoints at func

Watchpoints
-----------

Watchpoints pause execution when a variable's value changes, useful for tracking
down unexpected modifications to data.

.. code-block:: bash

    (gdb) watch var              # break when var changes
    (gdb) watch *0x7fff5678      # watch memory address
    (gdb) watch expr             # break when expression changes
    (gdb) rwatch var             # break when var is read
    (gdb) awatch var             # break on read or write
    (gdb) info watchpoints       # list watchpoints

Examining Variables
-------------------

Inspect variable values and types during debugging:

.. code-block:: bash

    (gdb) print var              # print variable value (p)
    (gdb) print/x var            # print in hexadecimal
    (gdb) print/t var            # print in binary
    (gdb) print arr[0]@10        # print 10 elements of array
    (gdb) print *ptr             # dereference pointer
    (gdb) print sizeof(var)      # print size

    (gdb) ptype var              # print type of variable
    (gdb) whatis var             # print type (brief)
    (gdb) info locals            # print all local variables
    (gdb) info args              # print function arguments

    (gdb) display var            # print var after each step
    (gdb) undisplay 1            # stop displaying

Examining Memory
----------------

The ``x`` command examines raw memory. Format: ``x/NFU address`` where N is
count, F is format (x=hex, d=decimal, s=string, i=instruction), and U is unit
size (b=byte, h=halfword, w=word, g=giant/8-byte).

.. code-block:: bash

    (gdb) x/s str                # print as string
    (gdb) x/10c str              # print 10 characters
    (gdb) x/4xw &var             # print 4 words in hex
    (gdb) x/10i $pc              # print 10 instructions at PC
    (gdb) x/20xb buf             # print 20 bytes in hex

Example with a character array:

.. code-block:: bash

    (gdb) x/s arr
    0x7fffffffe620: "Hello, World!"
    (gdb) x/5c arr
    0x7fffffffe620: 72 'H'  101 'e' 108 'l' 108 'l' 111 'o'
    (gdb) x/5xb arr
    0x7fffffffe620: 0x48    0x65    0x6c    0x6c    0x6f

Stack Traces
------------

Examine the call stack to understand how execution reached the current point:

.. code-block:: bash

    (gdb) backtrace              # show call stack (bt)
    (gdb) backtrace full         # show stack with local variables
    (gdb) backtrace 5            # show only 5 frames
    (gdb) where                  # same as backtrace

    (gdb) frame 2                # select frame 2 (f 2)
    (gdb) up                     # move up one frame
    (gdb) down                   # move down one frame
    (gdb) info frame             # detailed frame info

Debugging Crashes
-----------------

When a program crashes with a segmentation fault, GDB helps identify the cause:

.. code-block:: bash

    $ gdb ./myprogram
    (gdb) run
    Program received signal SIGSEGV, Segmentation fault.
    0x0000555555555149 in func () at main.c:10
    10          *ptr = 42;
    (gdb) bt
    #0  0x0000555555555149 in func () at main.c:10
    #1  0x0000555555555180 in main () at main.c:20
    (gdb) print ptr
    $1 = (int *) 0x0

Debug a core dump after a crash:

.. code-block:: bash

    $ ulimit -c unlimited        # enable core dumps
    $ ./myprogram                # run and crash
    $ gdb ./myprogram core       # analyze core dump
    (gdb) bt                     # see where it crashed

Reverse Debugging
-----------------

GDB can record execution and step backwards, invaluable for finding the cause
of bugs that manifest later than their origin.

.. code-block:: bash

    (gdb) record                 # start recording
    (gdb) continue               # run program
    (gdb) reverse-next           # step backwards (rn)
    (gdb) reverse-step           # step backwards into functions (rs)
    (gdb) reverse-continue       # run backwards to previous breakpoint
    (gdb) record stop            # stop recording

Multi-threaded Debugging
------------------------

Debug programs with multiple threads:

.. code-block:: bash

    (gdb) info threads           # list all threads
    (gdb) thread 2               # switch to thread 2
    (gdb) thread apply all bt    # backtrace all threads
    (gdb) set scheduler-locking on   # only run current thread

Useful Settings
---------------

Customize GDB behavior:

.. code-block:: bash

    (gdb) set print pretty on    # format structs nicely
    (gdb) set print array on     # format arrays nicely
    (gdb) set pagination off     # don't pause output
    (gdb) set logging on         # log output to gdb.txt
    (gdb) set history save on    # save command history

Define custom commands in ``~/.gdbinit``:

.. code-block:: bash

    # ~/.gdbinit
    set print pretty on
    set pagination off

    define sf
      where
      info args
      info locals
    end


CUDA Debugging with cuda-gdb
----------------------------

NVIDIA's cuda-gdb extends GDB for debugging CUDA kernels running on the GPU.
It supports breakpoints in device code, inspecting GPU threads and memory,
and switching between CPU and GPU contexts.

Compile with debug symbols for both host and device:

.. code-block:: bash

    $ nvcc -g -G -o myprogram main.cu   # -g for host, -G for device
    $ cuda-gdb ./myprogram

Basic CUDA debugging commands:

.. code-block:: bash

    (cuda-gdb) break main                # breakpoint in host code
    (cuda-gdb) break mykernel            # breakpoint in kernel
    (cuda-gdb) run

    (cuda-gdb) info cuda threads         # list all GPU threads
    (cuda-gdb) info cuda kernels         # list active kernels
    (cuda-gdb) info cuda blocks          # list thread blocks
    (cuda-gdb) info cuda lanes           # list lanes in current warp

Switch between GPU threads:

.. code-block:: bash

    (cuda-gdb) cuda thread               # show current thread
    (cuda-gdb) cuda thread 0 0 0         # switch to thread (0,0,0)
    (cuda-gdb) cuda block 1 0 0          # switch to block (1,0,0)
    (cuda-gdb) cuda kernel 0             # switch to kernel 0

    # Thread coordinates: (threadIdx.x, threadIdx.y, threadIdx.z)
    # Block coordinates: (blockIdx.x, blockIdx.y, blockIdx.z)

Inspect GPU memory and variables:

.. code-block:: bash

    (cuda-gdb) print threadIdx.x         # print thread index
    (cuda-gdb) print blockIdx.x          # print block index
    (cuda-gdb) print blockDim.x          # print block dimensions
    (cuda-gdb) print arr[threadIdx.x]    # print device array element

    (cuda-gdb) info cuda devices         # list GPU devices
    (cuda-gdb) info cuda sms             # list streaming multiprocessors

Conditional breakpoints in kernels:

.. code-block:: bash

    # Break only for specific thread
    (cuda-gdb) break mykernel if threadIdx.x == 0 && blockIdx.x == 0

    # Break when variable has specific value
    (cuda-gdb) break kernel.cu:42 if result > 100

Memory checking with cuda-memcheck:

.. code-block:: bash

    $ cuda-memcheck ./myprogram          # detect memory errors
    $ cuda-memcheck --tool racecheck ./prog   # detect race conditions
    $ cuda-memcheck --tool initcheck ./prog   # detect uninitialized memory
