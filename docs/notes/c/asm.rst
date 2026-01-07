=======================
X86 Assembly Reference
=======================

.. meta::
   :description: X86 and ARM64 assembly reference covering syscalls, inline assembly, CPUID, calling conventions, and common patterns for systems programming.
   :keywords: x86 assembly, ARM64 assembly, inline assembly, syscall, CPUID, calling convention, GCC asm, systems programming

.. contents:: Table of Contents
    :backlinks: none

Introduction
------------

Assembly language provides direct control over CPU instructions and is essential
for understanding how compilers generate code, writing performance-critical
routines, and interfacing with hardware at the lowest level. While high-level
languages abstract away machine details, assembly exposes the true nature of
computation: registers, memory addressing, instruction encoding, and the
hardware/software interface. This reference covers both standalone assembly
programs using Linux syscalls and GCC inline assembly for embedding assembly
in C code, with examples for x86_64 and ARM64 architectures. Understanding
assembly is fundamental for systems programming, reverse engineering, security
research, compiler development, and performance optimization where every cycle
counts.

Registers
---------

Registers are the fastest storage locations in a CPU, used for arithmetic
operations, function arguments, return values, and temporary data. The x86_64
architecture provides 16 general-purpose 64-bit registers, each accessible in
smaller widths (32-bit, 16-bit, 8-bit) for backward compatibility and efficient
storage. The System V AMD64 ABI defines conventions for which registers hold
function arguments, return values, and which must be preserved across function
calls (callee-saved vs caller-saved).

x86_64 general-purpose registers and their conventional uses:

.. code-block:: text

    Register  64-bit  32-bit  16-bit  8-bit   Purpose
    --------  ------  ------  ------  -----   -------
    RAX       rax     eax     ax      al      Return value, accumulator
    RBX       rbx     ebx     bx      bl      Callee-saved
    RCX       rcx     ecx     cx      cl      4th argument, counter
    RDX       rdx     edx     dx      dl      3rd argument, I/O
    RSI       rsi     esi     si      sil     2nd argument, source index
    RDI       rdi     edi     di      dil     1st argument, dest index
    RBP       rbp     ebp     bp      bpl     Base pointer (callee-saved)
    RSP       rsp     esp     sp      spl     Stack pointer
    R8-R15    r8-r15  r8d-r15d r8w-r15w r8b-r15b  5th-6th args, callee-saved (r12-r15)

ARM64 registers:

.. code-block:: text

    Register  64-bit  32-bit  Purpose
    --------  ------  ------  -------
    X0-X7     x0-x7   w0-w7   Arguments and return values
    X8        x8      w8      Indirect result, syscall number
    X9-X15    x9-x15  w9-w15  Temporary (caller-saved)
    X16-X17   x16-x17 w16-w17 Intra-procedure scratch
    X18       x18     w18     Platform register
    X19-X28   x19-x28 w19-w28 Callee-saved
    X29       x29/fp  w29     Frame pointer
    X30       x30/lr  w30     Link register (return address)
    SP        sp      wsp     Stack pointer

System Calls
------------

:Source: `src/asm/exit <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/asm/exit>`_

System calls are the fundamental interface between user-space programs and the
operating system kernel. When a program needs to perform privileged operations
like file I/O, memory allocation, or process management, it must request these
services from the kernel via syscalls. On x86_64 Linux, the ``syscall``
instruction invokes 64-bit system calls with the syscall number in ``rax`` and
up to six arguments in ``rdi``, ``rsi``, ``rdx``, ``r10``, ``r8``, ``r9``. The
return value appears in ``rax``, with negative values indicating errors (the
negated errno). The legacy ``int $0x80`` instruction invokes 32-bit syscalls
using different syscall numbers and the older convention of arguments in
``ebx``, ``ecx``, ``edx``, ``esi``, ``edi``, ``ebp``.

x86_64 exit (64-bit syscall):

.. code-block:: asm

    # gcc -nostdlib -no-pie -o exit64 exit64.s

    .global _start
    .section .text

    _start:
        mov $60, %rax       # syscall: exit (64-bit)
        mov $42, %rdi       # exit code
        syscall

x86 exit (32-bit syscall via int 0x80):

.. code-block:: asm

    # gcc -nostdlib -no-pie -m32 -o exit32 exit32.s

    .global _start
    .section .text

    _start:
        mov $1, %eax        # syscall: exit (32-bit)
        mov $42, %ebx       # exit code
        int $0x80

ARM64 exit:

.. code-block:: asm

    # gcc -nostdlib -o exit exit.s

    .global _start
    .section .text

    _start:
        mov x8, #93         // syscall: exit
        mov x0, #42         // exit code
        svc #0

Hello World
-----------

:Source: `src/asm/hello <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/asm/hello>`_

The classic "Hello World" program demonstrates essential assembly concepts: making
system calls, working with string data, and program termination. This example uses
the ``write`` syscall (number 1 on x86_64) to output text to stdout (file descriptor
1), then exits cleanly. The ``lea`` (load effective address) instruction with
RIP-relative addressing (``message(%rip)``) loads the address of the message string
in a position-independent way, which is required for modern executables. Data is
placed in the ``.rodata`` section for read-only constants, keeping code and data
properly separated.

.. code-block:: asm

    # gcc -nostdlib -no-pie -o hello hello.s

    .global _start
    .section .text

    _start:
        # write(1, message, 13)
        mov $1, %rax            # syscall: write
        mov $1, %rdi            # fd: stdout
        lea message(%rip), %rsi # buf: message (RIP-relative)
        mov $13, %rdx           # count
        syscall

        # exit(0)
        mov $60, %rax           # syscall: exit
        xor %rdi, %rdi          # status: 0
        syscall

    .section .rodata
    message:
        .ascii "Hello World\n"

.. code-block:: console

    $ ./hello
    Hello World

Loops
-----

:Source: `src/asm/loop <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/asm/loop>`_

Loop constructs in assembly use conditional jumps based on CPU flags set by
comparison or arithmetic instructions. The ``cmp`` instruction subtracts operands
and sets flags without storing the result, while ``dec`` decrements and sets the
zero flag when the result is zero. Jump instructions like ``jnz`` (jump if not
zero), ``je`` (jump if equal), and ``jl`` (jump if less) branch based on these
flags. When making syscalls inside loops, use callee-saved registers (``r12-r15``,
``rbx``, ``rbp``) for loop counters since syscalls may clobber caller-saved
registers. Local labels starting with ``.`` (like ``.loop``) are scoped to the
current function, preventing name collisions.

.. code-block:: asm

    .global _start
    .section .text

    _start:
        mov $5, %r12            # counter = 5

    .loop:
        # write(1, message, 13)
        mov $1, %rax
        mov $1, %rdi
        lea message(%rip), %rsi
        mov $13, %rdx
        syscall

        dec %r12                # counter--
        jnz .loop               # if (counter != 0) goto loop

        # exit(0)
        mov $60, %rax
        xor %rdi, %rdi
        syscall

    .section .rodata
    message:
        .ascii "Hello World\n"

Procedures
----------

:Source: `src/asm/procedure <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/asm/procedure>`_

Functions in assembly follow the System V AMD64 ABI calling convention, which
defines how arguments are passed, values returned, and registers preserved. The
``call`` instruction pushes the return address (address of the next instruction)
onto the stack and jumps to the target function. The ``ret`` instruction pops
this return address and jumps back to the caller. Stack frames are established
by the standard prologue: push ``rbp`` to save the caller's base pointer, then
set ``rbp`` to ``rsp`` to create a fixed reference point for local variables and
parameters. The epilogue reverses this by popping ``rbp`` before returning.
Functions must preserve callee-saved registers (``rbx``, ``rbp``, ``r12-r15``)
and ensure the stack is 16-byte aligned before any ``call`` instruction.

.. code-block:: asm

    .global _start
    .section .text

    _start:
        call print_hello
        call print_hello

        mov $60, %rax
        xor %rdi, %rdi
        syscall

    print_hello:
        push %rbp               # save base pointer
        mov %rsp, %rbp          # set up stack frame

        mov $1, %rax
        mov $1, %rdi
        lea message(%rip), %rsi
        mov $13, %rdx
        syscall

        pop %rbp                # restore base pointer
        ret

    .section .rodata
    message:
        .ascii "Hello World\n"


Command Line Arguments
----------------------

:Source: `src/asm/args <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/asm/args>`_

When a Linux program starts at ``_start``, the kernel has already set up the
stack with command-line arguments and environment variables. The stack layout
places ``argc`` (argument count) at ``(%rsp)``, followed by the ``argv`` array
of pointers starting at ``8(%rsp)``, terminated by a NULL pointer, then the
``envp`` array of environment variable pointers. Each ``argv[i]`` points to a
null-terminated C string. This example demonstrates iterating through all
arguments, computing string lengths with a custom ``strlen`` function, and
printing each argument followed by a newline. The ``cmpb`` instruction compares
bytes, and indexed addressing ``(%rdi, %rax)`` accesses ``string[index]``.

.. code-block:: asm

    .global _start
    .section .text

    _start:
        mov (%rsp), %r12        # r12 = argc
        lea 8(%rsp), %r13       # r13 = &argv[0]

    .print_args:
        cmp $0, %r12
        je .done

        mov (%r13), %rdi        # current argv
        call strlen             # length in %rax

        mov %rax, %rdx          # count
        mov $1, %rax            # write
        mov $1, %rdi            # stdout
        mov (%r13), %rsi        # buf
        syscall

        # print newline
        mov $1, %rax
        mov $1, %rdi
        lea newline(%rip), %rsi
        mov $1, %rdx
        syscall

        add $8, %r13            # next argv
        dec %r12
        jmp .print_args

    .done:
        mov $60, %rax
        xor %rdi, %rdi
        syscall

    strlen:
        xor %rax, %rax
    .strlen_loop:
        cmpb $0, (%rdi, %rax)
        je .strlen_done
        inc %rax
        jmp .strlen_loop
    .strlen_done:
        ret

    .section .rodata
    newline: .ascii "\n"

GCC Inline Assembly
-------------------

:Source: `src/asm/inline-asm <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/asm/inline-asm>`_

GCC inline assembly embeds assembly instructions directly in C code, providing
access to CPU features, special instructions, and hardware interfaces not exposed
through standard C. The extended asm syntax uses a template string with operand
placeholders (``%0``, ``%1``, etc.) and specifies output operands, input operands,
and clobbered registers. This information allows the compiler to allocate registers,
schedule instructions, and optimize around the assembly block. The ``volatile``
keyword prevents the compiler from removing or reordering the assembly, essential
for operations with side effects like memory barriers or I/O. Inline assembly is
commonly used for atomic operations, SIMD intrinsics, reading performance counters,
and implementing lock-free data structures.

Basic syntax:

.. code-block:: c

    __asm__ [volatile] (
        "assembly template"
        : outputs      // "=constraint"(variable)
        : inputs       // "constraint"(expression)
        : clobbers     // registers modified
    );

Constraint letters:

.. code-block:: text

    =    Output operand (write-only)
    +    Output operand (read-write)
    r    General register
    a    rax/eax/ax/al
    b    rbx/ebx/bx/bl
    c    rcx/ecx/cx/cl
    d    rdx/edx/dx/dl
    m    Memory operand
    i    Immediate integer

Read timestamp counter:

.. code-block:: c

    static inline uint64_t rdtsc(void) {
        uint32_t lo, hi;
        __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
        return ((uint64_t)hi << 32) | lo;
    }

Memory barrier:

.. code-block:: c

    // x86_64
    static inline void mfence(void) {
        __asm__ volatile ("mfence" ::: "memory");
    }

    // ARM64
    static inline void dmb(void) {
        __asm__ volatile ("dmb ish" ::: "memory");
    }

Byte swap:

.. code-block:: c

    static inline uint32_t bswap32(uint32_t x) {
        __asm__ ("bswapl %0" : "+r"(x));
        return x;
    }

CPUID
-----

:Source: `src/asm/cpuid <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/asm/cpuid>`_

The ``cpuid`` instruction is the standard mechanism for querying x86 CPU
identification and feature information at runtime. It takes a "leaf" number in
``eax`` (and optionally a "subleaf" in ``ecx``) and returns information in all
four registers (``eax``, ``ebx``, ``ecx``, ``edx``). Leaf 0 returns the maximum
supported leaf and the vendor identification string ("GenuineIntel" or
"AuthenticAMD"). Leaf 1 returns processor version information and feature flags
indicating support for SSE, AVX, AES-NI, and other extensions. Extended leaves
(0x80000002-0x80000004) return the processor brand string. Runtime feature
detection using CPUID enables writing code that adapts to available CPU
capabilities, using optimized paths when advanced instructions are available.

.. code-block:: c

    typedef struct { uint32_t eax, ebx, ecx, edx; } cpuid_t;

    static inline cpuid_t cpuid(uint32_t leaf, uint32_t subleaf) {
        cpuid_t r;
        __asm__ volatile (
            "cpuid"
            : "=a"(r.eax), "=b"(r.ebx), "=c"(r.ecx), "=d"(r.edx)
            : "a"(leaf), "c"(subleaf)
        );
        return r;
    }

    void get_vendor(char *vendor) {
        cpuid_t id = cpuid(0, 0);
        memcpy(vendor + 0, &id.ebx, 4);
        memcpy(vendor + 4, &id.edx, 4);
        memcpy(vendor + 8, &id.ecx, 4);
        vendor[12] = '\0';
    }

Feature detection:

.. code-block:: c

    cpuid_t f = cpuid(1, 0);
    int has_sse42 = (f.ecx >> 20) & 1;
    int has_avx   = (f.ecx >> 28) & 1;
    int has_aesni = (f.ecx >> 25) & 1;

    cpuid_t e = cpuid(7, 0);
    int has_avx2  = (e.ebx >> 5) & 1;

Syscall Wrappers
----------------

:Source: `src/asm/syscall-wrapper <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/asm/syscall-wrapper>`_

Inline assembly enables creating lightweight syscall wrappers that invoke system
calls directly without going through the C library. This is useful for minimal
runtime environments, static binaries without libc, or when you need precise
control over the syscall interface. The wrappers must follow the kernel's syscall
ABI: syscall number in ``rax``, arguments in ``rdi``, ``rsi``, ``rdx``, ``r10``,
``r8``, ``r9``, and return value in ``rax``. The ``syscall`` instruction clobbers
``rcx`` (saved return address) and ``r11`` (saved RFLAGS), which must be declared
in the clobber list. The ``"memory"`` clobber tells the compiler that the syscall
may read or write arbitrary memory.
useful for minimal runtime environments or when libc is unavailable. The
``syscall`` instruction clobbers ``rcx`` and ``r11`` in addition to the
return value in ``rax``.

.. code-block:: c

    static inline long syscall3(long n, long a1, long a2, long a3) {
        long ret;
        __asm__ volatile (
            "syscall"
            : "=a"(ret)
            : "a"(n), "D"(a1), "S"(a2), "d"(a3)
            : "rcx", "r11", "memory"
        );
        return ret;
    }

    // Usage
    #define SYS_write 1
    syscall3(SYS_write, 1, (long)"Hello\n", 6);

Calling Convention Summary
--------------------------

Calling conventions define the contract between caller and callee for passing
arguments, returning values, and preserving registers. The System V AMD64 ABI,
used by Linux, macOS, and BSD, passes the first six integer/pointer arguments in
registers for efficiency, with additional arguments on the stack. Floating-point
arguments use XMM registers. Caller-saved registers may be overwritten by any
function call, so the caller must save them if needed. Callee-saved registers
must be preserved by functions that use them, typically by pushing them in the
prologue and popping in the epilogue. The stack must be 16-byte aligned before
any ``call`` instruction to ensure proper alignment for SSE instructions.

System V AMD64 ABI (Linux, macOS, BSD):

.. code-block:: text

    Arguments:     rdi, rsi, rdx, rcx, r8, r9 (then stack)
    Return:        rax (and rdx for 128-bit)
    Caller-saved:  rax, rcx, rdx, rsi, rdi, r8-r11
    Callee-saved:  rbx, rbp, r12-r15
    Stack:         16-byte aligned before call

Syscall convention (Linux x86_64):

.. code-block:: text

    Syscall #:     rax
    Arguments:     rdi, rsi, rdx, r10, r8, r9
    Return:        rax (negative = -errno)
    Clobbered:     rcx, r11

Reference
---------

- `Linux System Call Table <https://chromium.googlesource.com/chromiumos/docs/+/master/constants/syscalls.md>`_
- `System V AMD64 ABI <https://refspecs.linuxbase.org/elf/x86_64-abi-0.99.pdf>`_
- `GCC Inline Assembly HOWTO <https://www.ibiblio.org/gferg/ldp/GCC-Inline-Assembly-HOWTO.html>`_
- `ARM64 Instruction Set <https://developer.arm.com/documentation/ddi0596/latest>`_
