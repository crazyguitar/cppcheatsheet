========================
C Concurrency cheatsheet
========================

.. contents:: Table of Contents
    :backlinks: none

How to write a UNIX daemon
--------------------------

.. code-block:: c

    #include <stdio.h>
    #include <stdlib.h>
    #include <unistd.h>
    #include <syslog.h>
    #include <sys/stat.h>

    int main(int argc, char *argv[])
    {
        int ret = -1;
        pid_t pid;

        /* become daemon */

        pid = fork();
        if (-1 == pid) {
            printf("Fork get error\n");
            goto Error;
        } else if (pid != 0) {
            ret = 0;
            goto Error;
        }
        /* Change the file mode mask */
        umask(0);

        /* set sid */
        if (-1 == setsid()) {
            printf("set sid failed\n");
            goto Error;
        }
        /* chdir to root "/" */
        if (-1 == chdir("/")) {
            printf("chdir(\"/\") failed\n");
            goto Error;
        }
        /* close stdin, stdout, stderr */
        close(STDIN_FILENO);
        close(STDOUT_FILENO);
        close(STDERR_FILENO);

        /* Do some task here */
        while (1) { sleep(3); syslog(LOG_ERR, "Hello"); }

        ret = 0;
    Error:
        return ret;
    }

Using ``daemon(nochdir, noclose)``
----------------------------------

.. code-block:: c

    #include <stdio.h>
    #include <unistd.h>
    #include <syslog.h>

    int main(int argc, char *argv[])
    {
        int ret = -1;
        /* make process as a daemon */
        if (-1 == daemon(0, 0)) {
            syslog(LOG_ERR, "create a daemon get error");
            goto Error;
        }
        /* do the daemon task */
        while(1) { sleep(3); syslog(LOG_ERR, "Hello"); }
        ret = 0;
    Error:
        return ret;
    }
