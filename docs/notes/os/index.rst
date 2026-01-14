==================
System Programming
==================

.. meta::
   :description: System programming in C/C++ covering file I/O operations, network socket programming, signal handling, and concurrency primitives on Linux/Unix systems.
   :keywords: system programming, Linux, Unix, POSIX, file I/O, socket programming, TCP/IP, signals, multithreading, concurrency, IPC

Building software that interacts directly with the operating system requires understanding
system calls, POSIX APIs, and the abstractions that Unix-like systems provide. System programming
bridges the gap between application logic and kernel services, enabling developers to build
efficient servers, daemons, and utilities.

Covered topics include file I/O operations and descriptor management, network communication
through BSD sockets supporting TCP/IP and UDP protocols, signal handling for asynchronous events,
and concurrent execution using threads and synchronization primitives. Mastery of these concepts
is essential for developing robust network services and high-performance applications.

.. toctree::
   :maxdepth: 1

   os_process
   os_file
   os_socket
   os_signal
   os_thread
   os_ipc
