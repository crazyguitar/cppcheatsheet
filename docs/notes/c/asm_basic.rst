=======================
X86 Assembly cheatsheet
=======================

.. contents:: Table of Contents
    :backlinks: none

Exit
----

.. code-block:: asm

    # gcc -o a.out -nostdlib  a.s

    .global _start
    .section .text

    _start:

    mov $0x1,%eax   # 32 bit of exit is 1
    mov $0x1,%ebx
    int $0x80

    .section .data

Note that ``int 0x80`` always invokes 32-bit system calls. To use system calls
define on X64 systems, we need to use ``syscall`` instruction.

.. code-block:: asm

    .global _start
    .section .text

    _start:

    mov $0x3c,%eax   # 64 bit of exit is 60(0x3c)
    mov $0x1,%ebx
    syscall

    .section .data

Hello Word
----------

.. code-block:: asm

    # gcc -o a.out -nostdlib  a.s
    # ./a.out
    # Hello World

    .global _start
    .section .text

    _start:

    # write(stdout, "Hello World", 13);

    mov $0x4,%eax       # 32 bit write syscall number
    mov $0x1,%ebx       # unsigned int fd (stdout)
    lea (message),%ecx  # const char *buf
    mov $13,%edx        # size_t count
    int $0x80

    # exit(0)

    mov $0x1,%eax
    mov $0x0,%ebx
    int $0x80

    .section .data
    message:
    .ascii "Hello World\n"

do while
--------

.. code-block:: asm

    .global _start
    .section .text

    _start:

    mov $0x1,%rsi

    loop: # do {

    # write(stdout, "Hello World\n", 13)
    mov $0x4,%eax
    mov $0x1,%ebx
    lea (message),%ecx
    mov $13,%edx
    int $0x80

    add $0x1,%rsi
    cmp $0x5,%rsi
    jbe loop  # } while(i<=5)

    # exit
    mov $0x1,%eax
    mov $0x0,%ebx
    int $0x80

    .section .data
    message: .ascii "Hello World\n"

Procedures
----------

.. code-block:: asm

    .global _start
    .section .text

    _start:

    callq print

    # exit
    mov $0x1,%eax
    mov $0x0,%ebx
    int $0x80

    print:
    # write(stdout, "Hello World\n", 13)
    mov $0x4,%eax
    mov $0x1,%ebx
    lea (message),%ecx
    mov $13,%edx
    int $0x80
    ret

    .section .data
    message: .ascii "Hello World\n"

Reference
---------

- `Linux System Call Table <https://chromium.googlesource.com/chromiumos/docs/+/master/constants/syscalls.md>`_
- `x86_64 Assembly Linux System Call Confusion <https://stackoverflow.com/q/8510333>`_
