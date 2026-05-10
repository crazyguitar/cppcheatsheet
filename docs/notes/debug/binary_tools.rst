============
Binary Tools
============

.. meta::
   :description: Guide to binary inspection tools nm, readelf, objdump, file, ldd, strings, and size for analyzing ELF executables, shared libraries, and object files.
   :keywords: nm, readelf, objdump, file, ldd, strings, size, ELF, symbol table, binary analysis, shared library, object file, C++ name mangling, c++filt

.. contents:: Table of Contents
    :backlinks: none

Introduction
------------

Binary inspection tools examine compiled executables, object files, and shared
libraries without running them. They reveal symbol tables, section layouts,
dependencies, and embedded strings—essential for debugging linker errors,
understanding library dependencies, and reverse engineering binaries.

On Linux, binaries use the ELF (Executable and Linkable Format) and tools like
``readelf``, ``objdump``, and ``ldd`` are the standard way to inspect them. On
macOS, binaries use the Mach-O format and the equivalent tool is ``otool``.
Some tools like ``nm``, ``strings``, ``size``, ``file``, ``strip``, and
``c++filt`` work on both platforms. When debugging "undefined reference" linker
errors, checking shared library dependencies, understanding why a binary is
large, or verifying that symbols are exported correctly, these tools are
indispensable.

:Source: `src/debug/binary-inspect <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/debug/binary-inspect>`_

All examples below use the ``binary-inspect_test`` executable built from
``src/debug/binary-inspect/``. The companion script ``inspect.sh`` exercises
every tool covered in this note against the compiled binary—run it standalone
or through ``ctest`` to verify the commands work on your system.

.. code-block:: bash

    # Build and run
    $ cmake -B build -S . && cmake --build build --target binary-inspect_test
    $ ./src/debug/binary-inspect/inspect.sh ./build/src/debug/binary-inspect/binary-inspect_test

Cheatsheet
----------

.. code-block:: bash

    # --- file: identify binary type & arch ---
    file ./binary
    file /usr/lib/x86_64-linux-gnu/libc.so.6

    # --- nm: list symbols ---
    nm ./binary                    # all symbols (mangled)
    nm -C ./binary                 # demangle C++ names
    nm -u ./binary                 # undefined symbols only
    nm -g ./binary                 # global (external) symbols only
    nm -n ./binary                 # sort by address
    nm -S ./binary                 # show symbol sizes
    nm -DC /usr/lib/libm.so.6     # dynamic symbols in shared lib

    # --- readelf: inspect ELF structure ---
    readelf -h ./binary            # ELF header (arch, entry point)
    readelf -S ./binary            # section headers (.text .data .bss ...)
    readelf -s ./binary            # symbol table
    readelf -l ./binary            # program headers (LOAD segments)
    readelf -d ./binary            # dynamic section (NEEDED libs)
    readelf -r ./binary            # relocations
    readelf -n ./binary            # notes (build-id, ABI)
    readelf -a ./binary            # all of the above
    readelf --debug-dump=info ./binary   # DWARF debug info
    readelf -p .rodata ./binary    # dump .rodata as strings

    # --- objdump: disassemble ---
    objdump -d -C ./binary         # disassemble with demangling
    objdump -d -S -C ./binary      # interleave source (needs -g)
    objdump -s -j .rodata ./binary # hex dump of .rodata section
    objdump -t -C ./binary         # symbol table
    objdump -R ./binary            # dynamic relocations

    # --- ldd: shared library dependencies ---
    ldd ./binary                   # list required .so files
    # WARNING: ldd runs the dynamic linker — don't use on untrusted binaries
    # Use readelf -d instead for untrusted files

    # --- strings: extract printable strings ---
    strings ./binary               # default min length 4
    strings -n 10 ./binary         # min length 10
    strings -t x ./binary          # show hex offset of each string
    strings ./binary | grep -i version

    # --- size: section sizes ---
    size ./binary                  # text, data, bss totals
    size -A ./binary               # all sections with sizes

    # --- strip: remove symbols ---
    strip ./binary                 # strip in place
    strip -o stripped ./binary     # strip to new file
    strip --strip-debug ./binary   # remove debug info only, keep symbols

    # --- c++filt: demangle ---
    echo "_ZN4demo6Widget5valueEv" | c++filt
    nm ./binary | c++filt | grep demo

    # --- ar: static library inspection ---
    ar -t libfoo.a                 # list object files in archive
    ar -x libfoo.a                 # extract all object files
    nm -C libfoo.a                 # list symbols across all members

    # --- otool: macOS equivalent of readelf/objdump/ldd ---
    otool -h ./binary              # Mach-O header (magic, cputype, filetype)
    otool -l ./binary              # load commands (segments, sections)
    otool -L ./binary              # shared library dependencies (like ldd)
    otool -tV ./binary             # disassemble __TEXT,__text
    otool -s __TEXT __cstring ./binary  # dump a specific section
    otool -p _add -tV ./binary     # disassemble starting at symbol _add

    # --- install_name_tool: macOS rpath/dylib management ---
    install_name_tool -change old.dylib new.dylib ./binary
    install_name_tool -add_rpath /usr/local/lib ./binary

nm: Symbol Table
----------------

``nm`` lists symbols in object files and executables. Each line shows an address,
a symbol type letter, and the symbol name. The type letter tells you where the
symbol lives: ``T`` for code in ``.text``, ``D`` for initialized data in
``.data``, ``B`` for zero-initialized data in ``.bss``, ``R`` for read-only
data in ``.rodata``, and ``U`` for undefined symbols that must be resolved by
the linker. Uppercase means the symbol is global (externally visible), lowercase
means it is local to the translation unit. This is the first tool to reach for
when debugging linker errors like "undefined reference" or "multiple definition".

Common symbol types:

.. code-block:: text

    T/t   Code (.text) — global/local
    D/d   Initialized data (.data) — global/local
    B/b   Uninitialized data (.bss) — global/local
    R/r   Read-only data (.rodata) — global/local
    U     Undefined (needs linking)
    W/w   Weak symbol

List all symbols:

.. code-block:: bash

    $ nm build/src/debug/binary-inspect/binary-inspect_test

C++ symbols are mangled. Use ``c++filt`` or ``-C`` to demangle:

.. code-block:: bash

    $ nm -C build/src/debug/binary-inspect/binary-inspect_test

Filter by symbol type:

.. code-block:: bash

    # Only undefined symbols (external dependencies)
    $ nm -u build/src/debug/binary-inspect/binary-inspect_test

    # Only defined symbols, sorted by address
    $ nm -n build/src/debug/binary-inspect/binary-inspect_test

    # Only global (external) symbols
    $ nm -g build/src/debug/binary-inspect/binary-inspect_test

Inspect an object file to see what it exports:

.. code-block:: bash

    $ nm -C build/src/debug/binary-inspect/CMakeFiles/binary-inspect_test.dir/helper.cc.o
    # Expected output:
    # 0000000000000000 D global_initialized
    # 0000000000000004 C global_uninitialized
    # 0000000000000000 T add(int, int)

readelf: ELF Structure
----------------------

``readelf`` displays detailed information about ELF (Executable and Linkable
Format) files—the standard binary format on Linux. Unlike ``objdump``, it does
not rely on the BFD library and can parse corrupted or partially linked files.
It is the most comprehensive tool for inspecting ELF metadata: the file header
reveals architecture and entry point, section headers show how the binary is
organized (``.text``, ``.data``, ``.bss``, ``.rodata``, etc.), program headers
describe how the kernel loads segments into memory, and the dynamic section
lists shared library dependencies. It also reads DWARF debug information, GNU
notes (build-id), and relocation entries. This is a Linux-only tool; on macOS,
use ``otool -l`` for similar load command inspection.

ELF header (architecture, entry point, type):

.. code-block:: bash

    $ readelf -h build/src/debug/binary-inspect/binary-inspect_test
    # Shows: Class (32/64-bit), OS/ABI, Type (EXEC/DYN), Entry point

Section headers (all sections in the binary):

.. code-block:: bash

    $ readelf -S build/src/debug/binary-inspect/binary-inspect_test
    # Shows: .text, .data, .bss, .rodata, .symtab, .strtab, etc.

Symbol table:

.. code-block:: bash

    $ readelf -s build/src/debug/binary-inspect/binary-inspect_test

    # With demangling (readelf doesn't have -C, pipe through c++filt)
    $ readelf -s build/src/debug/binary-inspect/binary-inspect_test | c++filt

Program headers (segments loaded at runtime):

.. code-block:: bash

    $ readelf -l build/src/debug/binary-inspect/binary-inspect_test
    # Shows LOAD, DYNAMIC, INTERP segments

Dynamic section (shared library dependencies):

.. code-block:: bash

    $ readelf -d build/src/debug/binary-inspect/binary-inspect_test
    # Shows NEEDED entries (libstdc++, libgtest, libc, etc.)

DWARF debug info (if compiled with ``-g``):

.. code-block:: bash

    $ readelf --debug-dump=info build/src/debug/binary-inspect/binary-inspect_test | head -50

objdump: Disassembly
--------------------

``objdump`` disassembles machine code back to assembly and displays section
contents. It is the go-to tool when you need to see what the compiler actually
generated—whether to verify an optimization kicked in, understand a crash at a
specific instruction address, or compare codegen between compiler flags. With
``-S`` and debug info (``-g``), it interleaves source lines with assembly,
making it easy to correlate C/C++ code with the generated instructions. It can
also dump raw section contents as hex (``-s -j .rodata``), which is useful for
inspecting string literals, vtables, or constant data embedded in the binary.
On macOS, ``otool -tV`` provides similar disassembly for Mach-O files, and
the LLVM ``objdump`` (shipped with Xcode) supports both formats.

Disassemble all code:

.. code-block:: bash

    $ objdump -d -C build/src/debug/binary-inspect/binary-inspect_test

Disassemble a specific function:

.. code-block:: bash

    $ objdump -d -C build/src/debug/binary-inspect/binary-inspect_test | \
        sed -n '/<add(int, int)>:/,/^$/p'

Interleave source with assembly (requires ``-g``):

.. code-block:: bash

    $ objdump -d -S -C build/src/debug/binary-inspect/binary-inspect_test

Display section contents as hex:

.. code-block:: bash

    # Show .rodata (read-only data like string literals)
    $ objdump -s -j .rodata build/src/debug/binary-inspect/binary-inspect_test

file: File Type
---------------

``file`` identifies file type, architecture, and linking mode by reading magic
bytes and metadata from the file header. It instantly tells you whether a binary
is 32-bit or 64-bit, statically or dynamically linked, stripped or not stripped,
and what architecture it targets (x86-64, ARM64, etc.). This is often the first
command to run when you receive an unfamiliar binary or need to confirm that a
cross-compilation produced the correct target format. It works on any file type,
not just executables—object files, shared libraries, core dumps, and even
scripts are all recognized:

.. code-block:: bash

    $ file build/src/debug/binary-inspect/binary-inspect_test
    # ELF 64-bit LSB pie executable, x86-64, version 1 (GNU/Linux),
    # dynamically linked, interpreter /lib64/ld-linux-x86-64.so.2,
    # with debug_info, not stripped

Key information: 32/64-bit, static/dynamic linking, stripped/not stripped.

ldd: Shared Library Dependencies
---------------------------------

``ldd`` lists shared libraries required at runtime and where the dynamic linker
resolves them on the filesystem. When a program fails to start with "error while
loading shared libraries" or "cannot open shared object file", ``ldd`` shows
exactly which ``.so`` file is missing and where the linker searched. It also
reveals version mismatches—if a binary was linked against ``libfoo.so.2`` but
only ``libfoo.so.1`` is installed, ``ldd`` will show "not found" for that entry.
On macOS, use ``otool -L`` instead to list dylib dependencies:

.. code-block:: bash

    $ ldd build/src/debug/binary-inspect/binary-inspect_test
    # linux-vdso.so.1 => (0x00007ffd...)
    # libgtest.so => /usr/lib/... (0x00007f...)
    # libstdc++.so.6 => /usr/lib/... (0x00007f...)
    # libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f...)
    # /lib64/ld-linux-x86-64.so.2 (0x00007f...)

.. warning::

    ``ldd`` executes the binary's dynamic linker, so never use it on untrusted
    binaries. Use ``readelf -d`` or ``objdump -p`` instead for untrusted files.

strings: Embedded Strings
-------------------------

``strings`` scans a binary file and extracts sequences of printable characters
(by default, 4 or more consecutive printable bytes). This is surprisingly useful
in practice: you can find hardcoded file paths, configuration values, error
messages, version strings, URLs, SQL queries, and even accidentally embedded
credentials. It works on any file—not just executables—so you can also use it
on core dumps, firmware images, or unknown binary blobs to get a quick sense of
what they contain:

.. code-block:: bash

    $ strings build/src/debug/binary-inspect/binary-inspect_test | grep "binary-inspect"
    # binary-inspect v1.0

    # Minimum string length of 10 characters
    $ strings -n 10 build/src/debug/binary-inspect/binary-inspect_test

    # Show file offset of each string
    $ strings -t x build/src/debug/binary-inspect/binary-inspect_test

Useful for finding hardcoded paths, version strings, error messages, and
configuration values embedded in binaries.

size: Section Sizes
-------------------

``size`` shows the size of each major section in an object file or executable,
giving a quick overview of how binary space is distributed between code
(``.text``), initialized data (``.data``), and zero-initialized data (``.bss``).
This is the fastest way to answer "why is my binary so large?"—if ``.text`` is
huge, you have a lot of code (possibly from heavy template instantiation or
inlining); if ``.data`` is large, you have big initialized globals or static
arrays; if ``.bss`` is large, you have large uninitialized buffers. The ``-A``
flag shows every section individually, which reveals additional contributors
like debug info, exception handling tables, and string tables:

.. code-block:: bash

    $ size build/src/debug/binary-inspect/binary-inspect_test
    #    text    data     bss     dec     hex filename
    #  ...       ...      ...     ...     ...  binary-inspect_test

    # Berkeley format (default) vs System V format
    $ size -A build/src/debug/binary-inspect/binary-inspect_test
    # Shows every section with size and address

Compare object files to see code size impact:

.. code-block:: bash

    $ size build/src/debug/binary-inspect/CMakeFiles/binary-inspect_test.dir/helper.cc.o
    $ size build/src/debug/binary-inspect/CMakeFiles/binary-inspect_test.dir/main.cc.o

strip: Remove Symbols
----------------------

``strip`` removes symbol table entries and debug information from a binary,
significantly reducing its size. Production deployments typically strip binaries
to save disk space and memory, and to make reverse engineering harder. A debug
build with ``-g`` can easily be 5–10x larger than its stripped counterpart.
You can strip everything (``strip binary``), strip only debug info while keeping
the symbol table (``strip --strip-debug``), or output to a separate file
(``strip -o``). A common workflow is to keep an unstripped copy for debugging
and deploy the stripped version—if a crash occurs, you can use the unstripped
binary with GDB to get full symbol and line information from the core dump:

.. code-block:: bash

    # Check size before
    $ ls -la build/src/debug/binary-inspect/binary-inspect_test

    # Strip (use a copy to preserve the original)
    $ cp build/src/debug/binary-inspect/binary-inspect_test /tmp/stripped_test
    $ strip /tmp/stripped_test

    # Check size after
    $ ls -la /tmp/stripped_test

    # Verify symbols are gone
    $ nm /tmp/stripped_test
    # nm: /tmp/stripped_test: no symbols

c++filt: Demangle C++ Symbols
------------------------------

``c++filt`` translates mangled C++ symbol names back to human-readable form.
C++ compilers encode function signatures into symbol names to support
overloading, namespaces, and templates—so ``_ZN4demo6Widget5valueEv`` becomes
``demo::Widget::value()``. Most tools like ``nm`` and ``objdump`` support ``-C``
for built-in demangling, but ``c++filt`` is useful as a standalone filter when
piping output from tools that don't have a demangle flag (like ``readelf``) or
when you encounter a mangled name in a log file or crash report:

.. code-block:: bash

    $ echo "_ZN4demo6Widget5valueEv" | c++filt
    # demo::Widget::value()

    $ nm build/src/debug/binary-inspect/binary-inspect_test | c++filt | grep demo
    # ... T demo::Widget::Widget(int)
    # ... T demo::Widget::value() const

Practical Examples
------------------

Debugging "undefined reference" linker errors:

.. code-block:: bash

    # Find which object file defines a symbol
    $ nm -C build/src/debug/binary-inspect/CMakeFiles/binary-inspect_test.dir/*.o | grep "T add"

    # Find which library provides a symbol
    $ nm -DC /usr/lib/x86_64-linux-gnu/libm.so.6 | grep " T sin"

Checking if a binary is stripped:

.. code-block:: bash

    $ file build/src/debug/binary-inspect/binary-inspect_test | grep -c "not stripped"

Finding why a binary is large:

.. code-block:: bash

    $ size -A build/src/debug/binary-inspect/binary-inspect_test | sort -k2 -n -r | head
