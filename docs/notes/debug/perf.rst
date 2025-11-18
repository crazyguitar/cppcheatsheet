===============
Perf Cheatsheet
===============

.. contents:: Table of Contents
    :backlinks: none


Perf Top
--------

.. code-block:: bash

    # show top symbols which CPU sample rate is 99 Hertz.
    perf top -F 99

    # show top process name and arguments
    perf top -ns comm,dso

    # count system call by process. refreshing 1 sec
    perf top -e raw_syscalls:sys_enter -ns comm -d 1
