====
File
====

.. meta::
   :description: C file I/O tutorial with examples for reading, writing, copying files, popen, FIFOs, lseek, fstat, file types, and directory traversal on Unix/Linux systems.
   :keywords: C file I/O, FIFO named pipe, popen, fopen fread fwrite, lseek example, fstat file size, copy file C, file types stat, nftw directory walk, Unix file operations, POSIX file API

.. contents:: Table of Contents
    :backlinks: none

Introduction
------------

File I/O is fundamental to Unix programming. The Unix philosophy of "everything
is a file" means that the same system calls—``open()``, ``read()``, ``write()``,
``close()``—work for regular files, devices, pipes, and sockets. This unified
interface simplifies programming and enables powerful composition of tools.

Unix provides two levels of file I/O: low-level system calls (``open``, ``read``,
``write``) that work with file descriptors, and the standard C library's buffered
I/O (``fopen``, ``fread``, ``fwrite``) that provides automatic buffering and
portability. Low-level calls offer more control and are necessary for certain
operations like file locking and memory mapping, while buffered I/O is more
convenient for text processing and sequential access.

File Size with ``lseek``
------------------------

:Source: `src/file/lseek-size <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/file/lseek-size>`_

The ``lseek()`` system call repositions the file offset for an open file
descriptor. By seeking to the end of the file (``SEEK_END``) and reading
the resulting offset, you can determine the file size. This technique works
for regular files but not for pipes, sockets, or some device files where
seeking is not supported.

.. code-block:: c

    #include <stdio.h>
    #include <fcntl.h>
    #include <unistd.h>

    int main(int argc, char *argv[]) {
      int fd = open(argv[1], O_RDONLY);
      if (fd < 0) return 1;

      off_t start = lseek(fd, 0, SEEK_SET);
      off_t end = lseek(fd, 0, SEEK_END);
      printf("File size: %lld bytes\n", (long long)(end - start));

      close(fd);
    }

.. code-block:: console

    $ echo "Hello" > hello.txt
    $ ./lseek-size hello.txt
    File size: 6 bytes

File Size with ``fstat``
------------------------

:Source: `src/file/fstat-size <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/file/fstat-size>`_

The ``fstat()`` system call retrieves metadata about an open file, including
its size, permissions, ownership, and timestamps. Unlike ``lseek()``, ``fstat()``
doesn't modify the file offset and provides additional information. The ``stat()``
variant works with a pathname instead of a file descriptor.

.. code-block:: c

    #include <stdio.h>
    #include <fcntl.h>
    #include <unistd.h>
    #include <sys/stat.h>

    int main(int argc, char *argv[]) {
      int fd = open(argv[1], O_RDONLY);
      if (fd < 0) return 1;

      struct stat st;
      fstat(fd, &st);
      printf("File size: %lld bytes\n", (long long)st.st_size);

      close(fd);
    }

.. code-block:: console

    $ echo "Hello" > hello.txt
    $ ./fstat-size hello.txt
    File size: 6 bytes

Copy File Contents
------------------

:Source: `src/file/copy-file <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/file/copy-file>`_

Copying a file involves reading from the source and writing to the destination
in a loop until all data is transferred. The buffer size affects performance:
larger buffers reduce system call overhead but use more memory. This example
also preserves the source file's permissions using ``fstat()`` to read them
and passing them to ``open()`` when creating the destination.

.. code-block:: c

    #include <stdio.h>
    #include <fcntl.h>
    #include <unistd.h>
    #include <sys/stat.h>

    #define BUF_SIZE 4096

    int main(int argc, char *argv[]) {
      int sfd = open(argv[1], O_RDONLY);
      struct stat st;
      fstat(sfd, &st);

      int dfd = open(argv[2], O_WRONLY | O_CREAT | O_TRUNC, st.st_mode);

      char buf[BUF_SIZE];
      ssize_t n;
      while ((n = read(sfd, buf, BUF_SIZE)) > 0)
        write(dfd, buf, n);

      close(sfd);
      close(dfd);
    }

.. code-block:: console

    $ echo "Hello World" > source.txt
    $ ./copy-file source.txt dest.txt
    $ diff source.txt dest.txt

Read Lines from File
--------------------

:Source: `src/file/read-lines <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/file/read-lines>`_

The ``getline()`` function reads an entire line from a stream, dynamically
allocating or reallocating the buffer as needed. It handles lines of any
length and includes the newline character in the result. The caller must
free the buffer when done. This is the preferred way to read lines in C
as it avoids buffer overflow vulnerabilities present in ``gets()`` and
the complexity of ``fgets()`` with unknown line lengths.

.. code-block:: c

    #include <stdio.h>
    #include <stdlib.h>

    int main(int argc, char *argv[]) {
      FILE *f = fopen(argv[1], "r");
      if (!f) return 1;

      char *line = NULL;
      size_t len = 0;
      while (getline(&line, &len, f) != -1)
        printf("%s", line);

      free(line);
      fclose(f);
    }

Read Entire File into Memory
----------------------------

:Source: `src/file/read-all <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/file/read-all>`_

For small to medium files, reading the entire contents into memory simplifies
processing. This pattern uses ``fseek()`` and ``ftell()`` to determine the file
size, allocates a buffer, then reads everything with a single ``fread()`` call.
Be cautious with large files as this approach can exhaust available memory.

.. code-block:: c

    #include <stdio.h>
    #include <stdlib.h>

    int main(int argc, char *argv[]) {
      FILE *f = fopen(argv[1], "r");
      if (!f) return 1;

      fseek(f, 0, SEEK_END);
      long size = ftell(f);
      rewind(f);

      char *buf = malloc(size + 1);
      fread(buf, 1, size, f);
      buf[size] = '\0';

      printf("%s", buf);

      free(buf);
      fclose(f);
    }

Check File Types
----------------

:Source: `src/file/file-type <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/file/file-type>`_

The ``stat()`` function returns file metadata including the file type encoded
in the ``st_mode`` field. Unix supports several file types: regular files,
directories, symbolic links, block devices, character devices, FIFOs (named
pipes), and sockets. The ``S_IFMT`` mask extracts the type bits, and macros
like ``S_ISREG()`` and ``S_ISDIR()`` provide convenient type checking.

.. code-block:: c

    #include <stdio.h>
    #include <sys/stat.h>

    int main(int argc, char *argv[]) {
      struct stat st;
      if (stat(argv[1], &st) < 0) return 1;

      switch (st.st_mode & S_IFMT) {
        case S_IFREG:  printf("Regular file\n"); break;
        case S_IFDIR:  printf("Directory\n"); break;
        case S_IFLNK:  printf("Symbolic link\n"); break;
        case S_IFBLK:  printf("Block device\n"); break;
        case S_IFCHR:  printf("Character device\n"); break;
        case S_IFIFO:  printf("FIFO/pipe\n"); break;
        case S_IFSOCK: printf("Socket\n"); break;
        default:       printf("Unknown\n");
      }
    }

.. code-block:: console

    $ ./file-type /etc/hosts
    Regular file
    $ ./file-type /usr
    Directory
    $ ./file-type /dev/null
    Character device

Directory Tree Walk
-------------------

:Source: `src/file/tree-walk <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/file/tree-walk>`_

The ``nftw()`` function (new file tree walk) recursively traverses a directory
tree, calling a user-provided callback for each entry. It handles the complexity
of opening directories, reading entries, and managing recursion depth. The
``FTW_DEPTH`` flag processes directory contents before the directory itself
(post-order), useful for operations like recursive deletion. ``FTW_PHYS``
prevents following symbolic links.

.. code-block:: c

    #define _GNU_SOURCE
    #include <stdio.h>
    #include <ftw.h>

    int callback(const char *path, const struct stat *sb,
                 int type, struct FTW *ftwbuf) {
      printf("%s\n", path);
      return 0;
    }

    int main(int argc, char *argv[]) {
      nftw(argv[1], callback, 64, FTW_DEPTH | FTW_PHYS);
    }

.. code-block:: console

    $ ./tree-walk .
    ./main.c
    ./a.out
    .

Pipe to Shell Command with ``popen``
------------------------------------

:Source: `src/file/popen-cmd <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/file/popen-cmd>`_

The ``popen()`` function creates a pipe to a shell command, returning a
``FILE*`` stream for reading the command's output or writing to its input.
This simplifies executing external commands and processing their results
without manually managing ``fork()``, ``exec()``, and ``pipe()``. Always
use ``pclose()`` to close the stream and retrieve the command's exit status.

.. code-block:: c

    #include <stdio.h>
    #include <sys/wait.h>

    int main(void) {
      FILE *fp = popen("ls -la", "r");
      if (!fp) return 1;

      char buf[256];
      while (fgets(buf, sizeof(buf), fp))
        printf("%s", buf);

      int status = pclose(fp);
      printf("Exit status: %d\n", WEXITSTATUS(status));
    }

.. code-block:: console

    $ ./popen-cmd
    total 32
    drwxr-xr-x  3 user user 4096 Jan  6 10:00 .
    -rwxr-xr-x  1 user user 8192 Jan  6 10:00 popen-cmd
    ...
    Exit status: 0

Named Pipe (FIFO)
-----------------

:Source: `src/file/fifo <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/file/fifo>`_

Named pipes (FIFOs) are similar to anonymous pipes but exist as filesystem
entries, allowing communication between unrelated processes. Create a FIFO
with ``mkfifo()`` or the ``mkfifo`` command. One process opens it for writing
and another for reading. The open blocks until both ends are connected.
FIFOs are useful for simple IPC without the complexity of sockets.

.. code-block:: c

    // writer.c
    #include <stdio.h>
    #include <fcntl.h>
    #include <unistd.h>
    #include <sys/stat.h>

    int main(void) {
      mkfifo("/tmp/myfifo", 0666);
      int fd = open("/tmp/myfifo", O_WRONLY);
      write(fd, "Hello FIFO", 11);
      close(fd);
    }

.. code-block:: c

    // reader.c
    #include <stdio.h>
    #include <fcntl.h>
    #include <unistd.h>

    int main(void) {
      int fd = open("/tmp/myfifo", O_RDONLY);
      char buf[128];
      read(fd, buf, sizeof(buf));
      printf("Received: %s\n", buf);
      close(fd);
      unlink("/tmp/myfifo");
    }

.. code-block:: console

    # Terminal 1
    $ ./writer

    # Terminal 2
    $ ./reader
    Received: Hello FIFO
