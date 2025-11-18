============================
C file operations cheatsheet
============================

.. contents:: Table of Contents
    :backlinks: none

Calculate file size via ``lseek``
---------------------------------

.. code-block:: c

    #include <stdio.h>
    #include <stdlib.h>
    #include <unistd.h>
    #include <sys/types.h>
    #include <sys/stat.h>
    #include <fcntl.h>

    int main(int argc, char *argv[])
    {
        int ret = -1;
        int fd = -1;
        size_t s_offset = 0;
        size_t e_offset = -1;
        char *path = NULL;

        if (argc != 2) {
            printf("Usage: PROG file\n");
            goto Error;
        }
        path = argv[1];
        if(0 > (fd = open(path,O_RDONLY))) {
            printf("open failed\n");
            goto Error;
        }
        if (-1 == (s_offset = lseek(fd, 0, SEEK_SET))) {
            printf("lseek error\n");
            goto Error;
        }
        if (-1 == (e_offset = lseek(fd, 0, SEEK_END))) {
            printf("lseek error\n");
            goto Error;
        }
        printf("File Size: %ld byte\n", e_offset - s_offset);
        ret = 0;
    Error:
        if (fd>=0) {
            close(fd);
        }
        return ret;
    }

output:

.. code-block:: console

    $ echo "Hello" > hello.txt
    $ ./a.out hello.txt
    File Size: 6 byte


Using ``fstat`` get file size
-----------------------------

.. code-block:: c

    #include <stdio.h>
    #include <sys/types.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>

    int main(int argc, char *argv[])
    {
        int ret = -1;
        int fd = -1;
        struct stat st = {0};
        char *path = NULL;

        if (argc != 2) {
            printf("Usage: PROG file\n");
            goto Error;
        }
        path = argv[1];
        /* using fstat */
        if (-1 == (fd = open(path, O_RDONLY))) {
            printf("open file get error\n");
            goto Error;
        }
        if (-1 == fstat(fd, &st)) {
            printf("fstat get error\n");
            goto Error;
        }
        printf("File Size: %lld byte\n", st.st_size);
        ret = 0;
    Error:
        if (fd>=0) {
            close(fd);
        }
        return ret;
    }

output:

.. code-block:: console

    $ echo "Hello" > hello.txt
    $ ./a.out hello.txt
    File Size: 6 byte


Copy all content of a file
--------------------------

.. code-block:: c

    #include <stdio.h>
    #include <sys/types.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>

    #define COPY_BUF_SIZE 1024

    int main(int argc, char *argv[])
    {
        int ret = -1;
        int sfd = -1, dfd = -1;
        mode_t perm = 0;
        char *src = NULL;
        char *dst = NULL;
        char buf[COPY_BUF_SIZE] = {0};
        size_t r_size = 0;
        struct stat st = {0};

        if (argc != 3) {
            printf("Usage: PROG src dst\n");
            goto Error;
        }

        /* open source */
        src = argv[1];
        if (-1 == (sfd = open(src, O_RDONLY))) {
            printf("open source fail\n");
            goto Error;
        }
        /* read source permission */
        if (-1 == (fstat(sfd, &st))) {
            printf("fstat file error\n");
            goto Error;
        }
        /* copy destination */
        dst = argv[2];
        perm = st.st_mode; /* set file permission */
        if (-1 == (dfd = open(dst, O_WRONLY | O_CREAT, perm))) {
            printf("open destination fail\n");
            goto Error;
        }
        while (0 < (r_size = read(sfd, buf, COPY_BUF_SIZE))) {
            if (r_size != write(dfd, buf, r_size)) {
                printf("copy file get error\n");
                goto Error;
            }
        }
        ret = 0;
    Error:
        if (sfd >= 0) {
            close(sfd);
        }
        if (dfd >= 0) {
            close(dfd);
        }
        return ret;
    }

output:

.. code-block:: console

    $ echo "Hello" > hello.txt
    $ ./a.out hello.txt hello_copy.txt
    $ diff hello.txt hello_copy.txt


Copy some bytes of content to a file
------------------------------------

.. code-block:: c

    #include <stdio.h>
    #include <stdlib.h>
    #include <sys/types.h>
    #include <sys/stat.h>
    #include <unistd.h>
    #include <fcntl.h>

    int main(int argc, char *argv[])
    {
        int ret = -1;
        int sfd = -1, dfd = -1;
        size_t s_offset = 0;
        size_t d_offset = -1;
        mode_t perm = 0;
        char *src = NULL;
        char *dst = NULL;
        struct stat st = {0};
        char buf[1024] = {0};
        size_t size = 0;
        size_t r_size = 0;

        if (argc != 4) {
            printf("Usage: PROG src dst bytes\n");
            goto Error;
        }
        /* open source file */
        src = argv[1];
        if (0 > (sfd = open(src, O_RDONLY))) {
            printf("open source file error\n");
            goto Error;
        }
        /* get source file permission */
        if (-1 == fstat(sfd, &st)) {
            printf("fstat fail\n");
            goto Error;
        }
        /* open dst file */
        dst = argv[2];
        perm = st.st_mode;
        if (0 > (dfd = open(dst, O_WRONLY | O_CREAT, perm))) {
            printf("open destination file error\n");
            goto Error;
        }
        if (-1 == (d_offset = lseek(dfd, 0, SEEK_END))) {
            printf("lseek get error\n");
            goto Error;
        }
        if (-1 == (s_offset = lseek(sfd, d_offset, SEEK_SET))) {
            printf("lseek get error\n");
            goto Error;
        }
        /* bytes */
        size = atoi(argv[3]);
        if (-1 == (r_size = read(sfd, buf, size))) {
            printf("read content fail\n");
            goto Error;
        }
        if (r_size != write(dfd, buf, r_size)) {
            printf("write content fail\n");
            goto Error;
        }
        ret = 0;
    Error:
        if (sfd >= 0) {
            close(sfd);
        }
        if (dfd >= 0) {
            close(dfd);
        }
        return ret;
    }

output:

.. code-block:: console

    $ echo "Hello" > hello.txt
    $ $ ./a.out hello.txt hello_copy.txt 3
    $ cat hello_copy.txt
    Hel$./a.out hello.txt hello_copy.txt 3
    $ cat hello_copy.txt
    Hello
    $ diff hello.txt hello_copy.txt


Get lines of a file
-------------------

.. code-block:: c

    // basic API: fopen, getline

    #include <stdio.h>
    #include <stdlib.h>

    int main(int argc, char *argv[])
    {
        int ret = -1;
        FILE *f = NULL;
        ssize_t read_size = 0;
        size_t len = 0;
        char *path = NULL;
        char *line = NULL;

        if (argc != 2) {
            printf("Usage: PROG file\n");
            goto Error;
        }

        path = argv[1];
        if (NULL == (f = fopen(path, "r"))) {
            printf("Read file error");
            goto Error;
        }

        while (-1 != getline(&line, &len, f)) {
            printf("%s\n", line);
        }
        ret = 0;
    Error:
        if (line) {
            free(line);
            line = NULL;
        }
        if (f) {
            fclose(f);
        }
        return ret;
    }

Get lines of a file via ``std::getline``
----------------------------------------

.. code-block:: cpp

    #include <iostream>
    #include <fstream>
    #include <sstream>
    #include <string>

    int main(int argc, char *argv[])
    {
        std::ifstream f(argv[1]);
        for (std::string line; std::getline(f, line);) {
            std::cout << line << "\n";
        }
    }

Read content into memory from a file
------------------------------------

.. code-block:: c

    // basick API: fopen, fseek, ftell, rewind, fread
    #include <stdio.h>
    #include <stdlib.h>

    int main(int argc, char *argv[])
    {
        int ret = -1;
        FILE *f = NULL;
        char *path = NULL;
        int size = 0;
        int read_size = 0;
        char *buffer = NULL;

        if (argc != 2) {
            printf("Usage: PROG file\n");
            goto Error;
        }

        path = argv[1];
        if (NULL == (f = fopen(path, "r"))) {
            printf("Read %s into memory fail\n", path);
            goto Error;
        }
        fseek(f, 0, SEEK_END);
        size = ftell(f);
        rewind(f);

        if (NULL == (buffer = (char *)calloc(size, sizeof(char)))) {
            printf("malloc file fail\n");
            goto Error;
        }

        read_size = fread(buffer, 1, size, f);
        if (read_size != size) {
            printf("fread %s fail\n", path);
            goto Error;
        }
        buffer[size-1] = '\0';
        printf("%s\n", buffer);
        ret = 0;
    Error:
        if (buffer) {
            free(buffer);
            buffer = NULL;
        }
        if (f) {
            fclose(f);
        }
        return ret;
    }

Check file types
----------------

.. code-block:: c

    #include <stdio.h>
    #include <string.h>
    #include <sys/stat.h>
    #include <sys/types.h>
    #include <unistd.h>

    int main(int argc, char *argv[])
    {
        int ret = -1;
        struct stat st;
        char *path = NULL;

        bzero(&st, sizeof(struct stat));

        if (argc != 2) {
            printf("Usage: PROG file\n");
            goto Error;
        }
        path = argv[1];
        if (-1 == stat(path, &st)) {
            printf("stat %s get error\n", path);
            goto Error;
        }
        /* check file type */
        switch (st.st_mode & S_IFMT) {
            case S_IFBLK: printf("Block device\n"); break;
            case S_IFCHR: printf("Character device\n"); break;
            case S_IFDIR: printf("Directory\n"); break;
            case S_IFIFO: printf("FIFO pipe\n"); break;
            case S_IFLNK: printf("Symbolic link\n"); break;
            case S_IFREG: printf("Regular file\n"); break;
            case S_IFSOCK: printf("Socket\n"); break;
            default: printf("Unknown\n");
        }
        ret = 0;
    Error:
        return ret;
    }

output:

.. code-block:: console

    $ ./a.out /etc/hosts
    Regular file
    $ ./a.out /usr
    Directory
    ./a.out /dev/tty.Bluetooth-Incoming-Port
    Character device


File tree walk
---------------

.. code-block:: c

    #define _GNU_SOURCE
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <errno.h>
    #include <ftw.h>

    #define CHECK_RET(ret, fmt, ...)        \
        do {                                \
            if (ret < 0) {                  \
                printf(fmt, ##__VA_ARGS__); \
                goto End;                   \
            }                               \
        } while(0)

    #define CHECK_NULL(ret, fmt, ...)       \
        do {                                \
            if (ret == NULL) {              \
                printf(fmt, ##__VA_ARGS__); \
                goto End;                   \
            }                               \
        } while(0)

    int callback(const char *fpath, const struct stat *sb, int typeflag, struct FTW *ftwbuf)
    {
        CHECK_NULL(fpath, "fpath cannot be NULL\n");
        printf("%s\n", fpath);
    End:
        return 0;
    }

    int main(int argc, char *argv[])
    {
        int ret = -1;
        char *path = NULL;

        if (argc != 2) {
            perror("Usage: PROG [dirpath]\n");
            goto End;
        }

        path = argv[1];
        ret = nftw(path, callback, 64, FTW_DEPTH | FTW_PHYS);
        CHECK_RET(ret, "nftw(%s) fail. [%s]", path, strerror(errno));
    End:
        return ret;
    }

output:

.. code-block:: console

    $ gcc tree_walk.c
    $ ./a.out .
    ./tree_walk.c
    ./a.out
    .
