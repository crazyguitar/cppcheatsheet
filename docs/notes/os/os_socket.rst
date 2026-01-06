===================
C socket cheatsheet
===================

.. contents:: Table of Contents
    :backlinks: none

Get host via ``gethostbyname``
--------------------------------

.. code-block:: c

    #include <stdio.h>
    #include <string.h>
    #include <netdb.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>

    int main(int argc, char *argv[])
    {
        int ret = -1, i = 0;
        struct hostent *h_ent = NULL;
        struct in_addr **addr_list = NULL;

        if (argc != 2) {
            printf("Usage: COMMAND [name]\n");
            goto End;
        }
        h_ent = gethostbyname(argv[1]);
        if (h_ent == NULL) {
            printf("gethostbyname fail. %s", hstrerror(h_errno));
            goto End;
        }

        printf("Host Name: %s\n", h_ent->h_name);
        addr_list = (struct in_addr **)h_ent->h_addr_list;
        for (i=0; addr_list[i] != NULL; i++) {
            printf("IP Address: %s\n", inet_ntoa(*addr_list[i]));
        }

        ret = 0;
    End:
        return ret;
    }

output:

.. code-block:: bash

    $ cc -g -Wall -o gethostbyname gethostbyname.c
    $ ./gethostbyname localhost
    Host Name: localhost
    IP Address: 127.0.0.1
    $ ./gethostbyname www.google.com
    Host Name: www.google.com
    IP Address: 74.125.204.99
    IP Address: 74.125.204.105
    IP Address: 74.125.204.147
    IP Address: 74.125.204.106
    IP Address: 74.125.204.104
    IP Address: 74.125.204.103


Transform host & network endian
--------------------------------

.. code-block:: c

    #include <stdio.h>
    #include <stdint.h>
    #include <arpa/inet.h>

    static union {
        uint8_t buf[2];
        uint16_t uint16;
    } endian = { {0x00, 0x3a} };

    #define LITTLE_ENDIANNESS ((char)endian.uint16 == 0x00)
    #define BIG_ENDIANNESS ((char)endian.uint16 == 0x3a)

    int main(int argc, char *argv[])
    {
        uint16_t host_short_val = 0x01;
        uint16_t net_short_val = 0;
        uint32_t host_long_val = 0x02;
        uint32_t net_long_val = 0;

        net_short_val  = htons(host_short_val);
        net_long_val   = htonl(host_long_val);
        host_short_val = htons(net_short_val);
        host_long_val  = htonl(net_long_val);

        if (LITTLE_ENDIANNESS) {
            printf("On Little Endian Machine:\n");
        } else {
            printf("On Big Endian Machine\n");
        }
        printf("htons(0x%x) = 0x%x\n", host_short_val, net_short_val);
        printf("htonl(0x%x) = 0x%x\n", host_long_val, net_long_val);

        host_short_val = htons(net_short_val);
        host_long_val  = htonl(net_long_val);

        printf("ntohs(0x%x) = 0x%x\n", net_short_val, host_short_val);
        printf("ntohl(0x%x) = 0x%x\n", net_long_val, host_long_val);
        return 0;
    }

output:

.. code-block:: bash

    # on little endian machine
    $ ./a.out
    On Little Endian Machine:
    htons(0x1) = 0x100
    htonl(0x2) = 0x2000000
    ntohs(0x100) = 0x1
    ntohl(0x2000000) = 0x2

    # on big endian machine
    $ ./a.out
    On Big Endian Machine
    htons(0x1) = 0x1
    htonl(0x2) = 0x2
    ntohs(0x1) = 0x1
    ntohl(0x2) = 0x2


Basic TCP socket server
-------------------------

.. code-block:: c

    #include <stdio.h>
    #include <string.h>
    #include <unistd.h>
    #include <sys/socket.h>
    #include <netinet/in.h>

    #define BUF_SIZE 1024
    #define isvalidsock(s) (s > 0 ? 1 : 0 )

    static int port = 5566;

    int main(int argc, char *argv[])
    {
        int ret = -1;
        int s = -1;
        int c = -1;
        socklen_t clen = 0;
        ssize_t len = 0;
        struct sockaddr_in s_addr;
        struct sockaddr_in c_addr;
        const int on = 1;
        char buf[BUF_SIZE] = {0};

        /* set socket host and port */
        bzero(&s_addr, sizeof(s_addr));
        s_addr.sin_family = AF_INET;
        s_addr.sin_addr.s_addr = htonl(INADDR_ANY);
        s_addr.sin_port = htons(port);

        /* create socket */
        s = socket(AF_INET, SOCK_STREAM, 0);
        if (!isvalidsock(s)) {
            printf("Create socket fail\n");
            goto Error;
        }
        /* set sockopt */
        if (0 > setsockopt(s, SOL_SOCKET,
                SO_REUSEADDR, &on, sizeof(on))) {
            printf("setsockopt fail\n");
            goto Error;
        }
        /* bind address and port */
        if (0 > bind(s, (struct sockaddr *) &s_addr,
                sizeof(s_addr))) {
            printf("bind socket fail\n");
            goto Error;
        }
        /* listen */
        if (0 > listen(s, 10)) {
            printf("listen fail\n");
            goto Error;
        }
        for(;;) {
            clen = sizeof(c_addr);
            c = accept(s, (struct sockaddr *)&c_addr, &clen);
            if (!isvalidsock(c)) {
                printf("accept error\n");
                continue;
            }
            bzero(buf, BUF_SIZE);
            if (0 > (len = recv(c, buf, BUF_SIZE-1, 0))) {
                close(c);
            }
            send(c, buf, BUF_SIZE-1, 0);
            close(c);
        }
        ret = 0
    Error:
        if (s >= 0){
            close(s);
        }
        return ret;
    }

output:

.. code-block:: console

    $ ./a.out &
    [1] 63546
    $ nc localhost 5566
    Hello Socket
    Hello Socket


Basic UDP socket server
------------------------

.. code-block:: c

    #include <stdio.h>
    #include <string.h>
    #include <errno.h>
    #include <sys/socket.h>
    #include <sys/types.h>
    #include <arpa/inet.h>
    #include <netinet/in.h>
    #include <unistd.h>

    #define EXPECT_GE(i, e, ...) \
      if (i < e) {__VA_ARGS__}

    #define EXPECT_SUCCESS(ret, fmt, ...) \
      EXPECT_GE(ret, 0, \
        printf(fmt, ##__VA_ARGS__); goto End;)

    #ifndef BUF_SIZE
    #define BUF_SIZE 1024
    #endif

    int main(int argc, char *argv[])
    {
        int ret = -1;
        int sockfd = -1;
        int port = 5566;
        char buf[BUF_SIZE] = {};
        struct sockaddr_in s_addr = {};
        struct sockaddr_in c_addr = {};
        socklen_t s_len = 0;

        /* create socket */
        sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        EXPECT_SUCCESS(sockfd, "create socket fail. %s\n", strerror(errno));


        /* set socket addr */
        bzero((char *) &s_addr, sizeof(s_addr));
        s_addr.sin_family = AF_INET;
        s_addr.sin_port = htons(port);
        s_addr.sin_addr.s_addr = htonl(INADDR_ANY);
        s_len = sizeof(c_addr);

        /* bind */
        ret = bind(sockfd, (struct sockaddr *)&s_addr, sizeof(s_addr));
        EXPECT_SUCCESS(ret, "bind fail. %s\n", strerror(errno));

        for(;;) {
            bzero(buf, sizeof(buf));
            ret = recvfrom(sockfd, buf, sizeof(buf), 0,
                           (struct sockaddr *)&c_addr, &s_len);
            EXPECT_GE(ret, 0, continue;);

            ret = sendto(sockfd, buf, ret, 0,
                         (struct sockaddr *)&c_addr, s_len);
        }

        ret = 0;
    End:
        if (sockfd >= 0) {
            close(sockfd);
        }
        return ret;
    }

output:

.. code-block:: bash

    $ cc -g -Wall -o udp_server udp_server.c
    $ ./udp_server &
    [1] 90190
    $ nc -u 192.168.55.66 5566
    Hello
    Hello
    UDP
    UDP


Event driven socket via ``select``
----------------------------------

.. code-block:: c

    #include <stdio.h>
    #include <string.h>
    #include <unistd.h>
    #include <sys/types.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <errno.h>

    #define BUF_SIZE 1024
    #define isvalidsock(s) (s > 0 ? 1 : 0)
    #define PORT 5566

    int socket_init(void)
    {
        struct sockaddr_in s_addr;
        int sfd = -1;
        int ret = -1;
        const int on = 1;

        bzero(&s_addr, sizeof(s_addr));
        s_addr.sin_family = AF_INET;
        s_addr.sin_addr.s_addr = htonl(INADDR_ANY);
        s_addr.sin_port = htons(PORT);

        sfd = socket(AF_INET, SOCK_STREAM, 0);
        if (!isvalidsock(sfd)) {
            printf("create socket error\n");
            goto Error;
        }
        if (0 > setsockopt(
                sfd, SOL_SOCKET,
                SO_REUSEADDR, &on, sizeof(on))) {
            printf("setsockopt error\n");
            goto Error;
        }
        if (0 > bind(sfd,
                    (struct sockaddr *)&s_addr,
                    sizeof(s_addr))) {
            printf("bind error\n");
            goto Error;
        }
        if (0 > listen(sfd, 10)) {
            printf("listen network error\n");
            goto Error;
        }
        ret = sfd;
    Error:
        if (ret == -1) {
            if (sfd >=0) {
                close(sfd);
            }
        }
        return ret;
    }

    int main(int argc, char *argv[])
    {
        int ret = -1;
        int sfd = -1;
        int cfd = -1;
        ssize_t len = 0;
        struct sockaddr_in c_addr;
        int i = 0;
        int rlen = 0;
        char buf[BUF_SIZE] = {0};
        socklen_t clen = 0;
        fd_set wait_set;
        fd_set read_set;

        if (-1 == (sfd = socket_init())) {
            printf("socket_init error\n");
            goto Error;
        }
        FD_ZERO(&wait_set);
        FD_SET(sfd, &wait_set);
        for (;;) {
            read_set = wait_set;
            if (0 > select(FD_SETSIZE, &read_set,
                           NULL, NULL, NULL)) {
                printf("select get error\n");
                goto Error;
            }
            for (i=0; i < FD_SETSIZE; i++) {
                if (!FD_ISSET(i, &read_set)) {
                    continue;
                }
                if (i == sfd) {
                    clen = sizeof(c_addr);
                    cfd = accept(sfd,
                        (struct sockaddr *)&c_addr, &clen);
                    if (!isvalidsock(cfd)) {
                        goto Error;
                    }
                    FD_SET(cfd, &wait_set);
                } else {
                    bzero(buf, BUF_SIZE);
                    if (0 > (rlen = read(i, buf, BUF_SIZE-1))) {
                        close(i);
                        FD_CLR (i, &wait_set);
                        continue;
                    }
                    if (0 > (rlen = write(i, buf, BUF_SIZE-1))) {
                        close(i);
                        FD_CLR (i, &wait_set);
                        continue;
                    }
                }
            }
        }
        ret = 0;
    Error:
        if (sfd >= 0) {
            FD_CLR(sfd, &wait_set);
            close(sfd);
        }
        return ret;
    }

output: (bash 1)

.. code-block:: console

    $ ./a.out &
    [1] 64882
    Hello
    Hello

output: (bash 2)

.. code-block:: console

    $ nc localhost 5566
    Socket
    Socket


socket with pthread
---------------------

.. code-block:: c

    #include <stdio.h>
    #include <string.h>
    #include <errno.h>
    #include <sys/socket.h>
    #include <unistd.h>
    #include <netinet/in.h>
    #include <sys/types.h>
    #include <arpa/inet.h>
    #include <pthread.h>

    #define EXPECT_GE(i, e, ...) \
        if (i < e) { __VA_ARGS__; }

    #define EXPECT_SUCCESS(ret, fmt, ...) \
        EXPECT_GE(ret, 0, printf(fmt, ##__VA_ARGS__); goto End)

    #define SOCKET(sockfd, domain, types, proto) \
        do { \
            sockfd = socket(domain, types, proto); \
            EXPECT_SUCCESS(sockfd, "create socket fail. %s", strerror(errno)); \
        } while(0)

    #define SETSOCKOPT(ret, sockfd, level, optname, optval) \
        do { \
            int opt = optval;\
            ret = setsockopt(sockfd, level, optname, &opt, sizeof(opt)); \
            EXPECT_SUCCESS(ret, "setsockopt fail. %s", strerror(errno)); \
        } while(0)

    #define BIND(ret, sockfd, addr, port) \
        do { \
            struct sockaddr_in s_addr = {}; \
            struct sockaddr sa = {}; \
            socklen_t len = 0; \
            ret = getsockname(sockfd, &sa, &len); \
            EXPECT_SUCCESS(ret, "getsockopt fail. %s", strerror(errno)); \
            s_addr.sin_family = sa.sa_family; \
            s_addr.sin_addr.s_addr = inet_addr(addr); \
            s_addr.sin_port = htons(port); \
            ret = bind(sockfd, (struct sockaddr *) &s_addr, sizeof(s_addr)); \
            EXPECT_SUCCESS(ret, "bind fail. %s", strerror(errno)); \
        } while(0)

    #define LISTEN(ret, sockfd, backlog) \
        do { \
            ret = listen(sockfd, backlog); \
            EXPECT_SUCCESS(ret, "listen fail. %s", strerror(errno)); \
        } while(0)


    #ifndef BUF_SIZE
    #define BUF_SIZE 1024
    #endif

    void *handler(void *p_sockfd)
    {
        int ret = -1;
        char buf[BUF_SIZE] = {};
        int c_sockfd = *(int *)p_sockfd;

        for (;;) {
            bzero(buf, sizeof(buf));
            ret = recv(c_sockfd, buf, sizeof(buf) - 1, 0);
            EXPECT_GE(ret, 0, break);
            send(c_sockfd, buf, sizeof(buf) - 1, 0);
        }
        EXPECT_GE(c_sockfd, 0, close(c_sockfd));
        pthread_exit(NULL);
    }

    int main(int argc, char *argv[])
    {
        int ret = -1, sockfd = -1, c_sockfd = -1;
        int port = 9527;
        char addr[] = "127.0.0.1";
        struct sockaddr_in c_addr = {};
        socklen_t clen = 0;
        pthread_t t;

        SOCKET(sockfd, AF_INET, SOCK_STREAM, 0);
        SETSOCKOPT(ret, sockfd, SOL_SOCKET, SO_REUSEADDR, 1);
        BIND(ret, sockfd, addr, port);
        LISTEN(ret, sockfd, 10);

        for(;;) {
            c_sockfd = accept(sockfd, (struct sockaddr *)&c_addr, &clen);
            EXPECT_GE(c_sockfd, 0, continue);
            ret = pthread_create(&t, NULL, handler, (void *)&c_sockfd);
            EXPECT_GE(ret, 0, close(c_sockfd); continue);
        }
    End:
        EXPECT_GE(sockfd, 0, close(sockfd));
        ret = 0;
        return ret;
    }

output:

.. code-block:: bash

    # console 1
    $ cc -g -Wall -c -o test.o test.c
    $ cc test.o -o test
    $ ./test &
    [1] 86601
    $ nc localhost 9527
    Hello
    Hello

    # console 2
    $ nc localhost 9527
    World
    World
