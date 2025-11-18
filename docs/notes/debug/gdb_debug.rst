Debug with GDB
==============

.. contents:: Table of Contents
    :backlinks: none

Load an Executable
------------------


Using GDB to debug requires it recognizes a program's debug symbols. By
compiling with ``-g`` option, GDB will understand what source code looks like
after loading an executable file:

.. code-block:: bash

    $ gcc -g -Wall -Werror foo.c # compile with -g option
    $ gdb ./a.out  # load all symbols of a.out into GDB


Text User Interface
-------------------

Text User Interface (TUI) allows developers to visualize source code and to
debug like using the Integrated Development Environment (IDE) to trace problems.
For a beginner, entering the TUI mode is more understandable than the command
line mode. The following key bindings are the most common usages for interacting
with TUI.

1. Ctrl x + a - Enter or leave the TUI mode
2. Ctrl x + o - Switch the active window
3. Ctrl x + 1 - Display one window (e.g., source code + GDB shell)
4. Ctrl x + 2 - Display two windows (e.g., source code + GDB shell + assembly)
5. Ctrl l - Refresh window


Basic Commands
--------------

**Start/Stop a program**

1. start - Run an executable file and stop at the beginning
2. run / r - Run an executable file until finish or stop at a breakpoint
3. step / s - Run a program step by step with entering a function
4. next / n - Run a program step by step without entering a function
5. continue / c - Run a program until finish or stop at a breakpoint
6. finish - Step out of the current function

**Set Breakpoints**

1. b line - Set a breakpoint at the given line in the current file
2. b file: line - Set a breakpoint at the given line in a given file
3. b ... if cond - Set a breakpoint when the condition is true
4. clear line - Delete a breakpoint at the given line in the current file
5. clear file: line - Delete a breakpoint at giving a line in a given file
6. info breakpoints - Display breakpoints status
7. enable breakpoints - Enable breakpoints
8. disable breakpoints - Disable breakpoints
9. watch cond - Set a watchpoint for inspecting a value


**Display Stack**

1. backtrace / bt - Display current stack
2. frame / f framenum - Select a frame and inspect its status
3. where - Display the current stack and the line

**Print Variables**

1. print / p var - Print value of the given variable
2. ptype var - Print type info of the given variable
3. info args - Print function arguments
4. info locals - Print all local variables

**Reverse Run**

1. record - Start recording each instruction step
2. record stop - Stop recording
3. rn - Reverse next
4. rs - Reverse step

.. code-block:: cpp

    int main(int argc, char *argv[]) {
        int out = 0;
        for (int i = 0; i < 10; ++i) {
            out = i * i;
        }
        return out;
    }

.. code-block:: bash

	(gdb) b main
	(gdb) r
	Starting program: /home/ubuntu/a.out

	Breakpoint 1, main (argc=21845, argv=0x0) at test.cc:2
	2       {
	(gdb) record
	...
	(gdb) n
	(gdb) p out
	$1 = 1
	(gdb) rn
	(gdb) rn
	(gdb) p out
	$2 = 0

**Define a Function**

GDB provides an original way for developers to define a customized function.
The following snippet shows how to define a function to display the information
of the current stack.

.. code-block:: bash

    (gdb) define sf
    Type commands for definition of "sf".
    End with a line saying just "end".
    >where
    >info args
    >info locals
    >end

Display Memory Contents
-----------------------

.. code-block:: cpp

    int main() {
        char arr[100] = "1234567890abcdefghijklmnopqrstuvwxyz";
        return 0;
    }

.. code-block:: bash

    (gdb) " x/[format] [address expression]
    (gdb) " x/[len][format] [address expression]
    (gdb) x/s arr
    0x7fffffffe620:	"1234567890abcdefghijklmnopqrstuvwxyz"
    (gdb) x/10c arr
    (gdb) x/5c arr
    0x7fffffffe620:	49 '1'	50 '2'	51 '3'	52 '4'	53 '5'
    (gdb) x/5b arr
    0x7fffffffe620:	0x31	0x32	0x33	0x34	0x35
