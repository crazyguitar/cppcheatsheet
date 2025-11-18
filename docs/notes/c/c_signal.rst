=============================
C signal operation cheatsheet
=============================

.. contents:: Table of Contents
    :backlinks: none

Print signal expression
-----------------------

.. code-block:: c

    #include <stdio.h>
    #include <signal.h>

    #define ARRAYLEN(arr) sizeof(arr) / sizeof((arr)[0])

    static int signo_arr[] = {
        SIGABRT , SIGALRM  , SIGBUS,
        SIGCHLD , SIGCONT  , SIGFPE,
        SIGHUP  , SIGILL   , SIGINT,
        SIGIO   , SIGKILL  , SIGPIPE,
        SIGPROF , SIGQUIT  , SIGSEGV,
        SIGSYS  , SIGTERM  , SIGTRAP,
        SIGTSTP , SIGTTIN  , SIGTTOU,
        SIGURG  , SIGVTALRM, SIGUSR1,
        SIGUSR2 , SIGXCPU  , SIGXFSZ
    };

    int main(int argc, char *argv[])
    {
        int i = 0;
        int signo = -1;
        char *msg = "SIGNAL";

        for (i=0; i < ARRAYLEN(signo_arr); i++) {
            signo = signo_arr[i];
            printf("Signal[%d]: %s\n", signo, sys_siglist[signo]);
        }

        return 0;
    }

output:

.. code-block:: console

    $ ./a.out
    Signal[6]: Abort trap
    Signal[14]: Alarm clock
    Signal[10]: Bus error
    Signal[20]: Child exited
    Signal[19]: Continued
    Signal[8]: Floating point exception
    Signal[1]: Hangup
    Signal[4]: Illegal instruction
    Signal[2]: Interrupt
    Signal[23]: I/O possible
    Signal[9]: Killed
    Signal[13]: Broken pipe
    Signal[27]: Profiling timer expired
    Signal[3]: Quit
    Signal[11]: Segmentation fault
    Signal[12]: Bad system call
    Signal[15]: Terminated
    Signal[5]: Trace/BPT trap
    Signal[18]: Suspended
    Signal[21]: Stopped (tty input)
    Signal[22]: Stopped (tty output)
    Signal[16]: Urgent I/O condition
    Signal[26]: Virtual timer expired
    Signal[30]: User defined signal 1
    Signal[31]: User defined signal 2
    Signal[24]: Cputime limit exceeded
    Signal[25]: Filesize limit exceeded


Basic signal event handler
--------------------------

.. code-block:: c

    #include <stdio.h>
    #include <string.h>
    #include <signal.h>
    #include <errno.h>
    #include <sys/types.h>
    #include <unistd.h>

    /** singal handler prototype :
     *
     *  type void (*sighandler_t) (int)
     */

    void sig_handler(int signo)
    {
        printf("[%d] Get signal: %s\n", getpid(), strsignal(signo));
    }

    int main(int argc, char *argv[])
    {
        int ret = -1;

        /* overwrite default signal handler */
        if (SIG_ERR == signal(SIGHUP, sig_handler)) {
            printf("Get error: %s\n", strerror(errno));
            goto Error;
        }
        if (SIG_ERR == signal(SIGINT, sig_handler)) {
            printf("Get error: %s\n", strerror(errno));
            goto Error;
        }
        if (SIG_ERR == signal(SIGALRM, sig_handler)) {
            printf("Get error: %s\n", strerror(errno));
            goto Error;
        }
        /* ignore signal */
        if (SIG_ERR == signal(SIGUSR1, SIG_IGN)) {
            printf("Get error: %s\n", strerror(errno));
            goto Error;
        }
        while(1) { sleep(3); }
        ret = 0;
    Error:
        return ret;
    }

output:

.. code-block:: console

    $ ./a.out
    ^C[54652] Get signal: Interrupt: 2
    [54652] Get signal: Hangup: 1
    [54652] Get signal: Alarm clock: 14


A pthread signal handler
------------------------

.. code-block:: c

    #include <stdio.h>
    #include <stdlib.h>
    #include <pthread.h>
    #include <errno.h>
    #include <signal.h>
    #include <unistd.h>

    static void *sig_thread(void *arg)
    {
        sigset_t *set = (sigset_t *)arg;
        int err = -1, signo = -1;

        for(;;) {
            if(0 != (err = sigwait(set, &signo))) {
                printf("sigwait error\n");
                goto Error;
            }
            printf("Get signal[%d]: %s\n",
                   signo, sys_siglist[signo]);
        }
    Error:
        return;
    }

    int main(int argc, char *argv[])
    {
        pthread_t thread;
        sigset_t sig_set;
        int err = -1;

        sigemptyset(&sig_set);
        sigaddset(&sig_set, SIGQUIT);
        sigaddset(&sig_set, SIGUSR1);
        /* set signal handler thread sigmask */
        err = pthread_sigmask(SIG_BLOCK, &sig_set, NULL)
        if(0 != err) {
            printf("set pthread_sigmask error\n");
            goto Error;
        }
        /* create signal thread */
        err = pthread_create(&thread, NULL,
                             &sig_thread, (void *)&sig_set))
        if (0 != err) {
            printf("create pthread error\n");
            goto Error;
        }

        pause();
    Error:
        return err;
    }

output:

.. code-block:: console

    $ ./a.out &
    [1] 21258
    $ kill -USR1 %1
    Get signal[10]: User defined signal 1
    $ kill -QUIT %1
    Get signal[3]: Quit
    $ kill -TERM %1
    [1]+  Terminated              ./a.out


Check child process alive
-------------------------

.. code-block:: c

    #include <stdio.h>
    #include <unistd.h>
    #include <signal.h>

    void handler(int signo)
    {
        pid_t pid = getpid();
        printf("[%i] Got signal[%d]: %s\n",
               pid, signo, sys_siglist[signo]);
    }

    int main(int argc, char *argv[])
    {
        int ret = -1;
        pid_t pid = -1;

        pid = fork();
        signal(SIGCHLD, handler);
        if (pid < 0) {
            printf("Fork failed\n");
            goto Error;
        } else if (pid == 0) {
            /* child */
            printf("Child[%i]\n", getpid());
            sleep(3);
        } else {
            printf("Parent[%i]\n", getpid());
            pause();
        }
        ret = 0;
    Error:
        return ret;
    }

.. code-block:: console

    $ ./a.out
    Parent[59113]
    Child[59114]
    [59113] Got signal[20]: Child exited


Basic sigaction usage
---------------------

.. code-block:: c

    #include <stdio.h>
    #include <signal.h>
    #include <sys/types.h>
    #include <unistd.h>

    void handler(int signo)
    {
        printf("Get Signal: %s\n",sys_siglist[signo]);
    }

    int main(int argc, char *argv[])
    {
        pid_t pid = -1;
        struct sigaction new_sa = {0};
        struct sigaction old_sa = {0};

        new_sa.sa_handler = handler;
        sigemptyset(&new_sa.sa_mask);
        new_sa.sa_flags = 0;

        pid = getpid();
        printf("Process PID: %i\n", pid);
        /* if signal not ignore, overwrite its handler */
        sigaction(SIGINT, NULL, &old_sa);
        if (old_sa.sa_handler != SIG_IGN) {
            sigaction(SIGINT, &new_sa, NULL);
        }

        sigaction(SIGHUP, NULL, &old_sa);
        if (old_sa.sa_handler != SIG_IGN) {
            sigaction(SIGHUP, &new_sa, NULL);
        }
        while (1) { sleep(3); }
        return 0;
    }

output:

.. code-block:: console

    # bash 1
    kill -1 57140
    kill -2 57140

    # bash 2
    $ ./a.out
    Process PID: 57140
    Get Signal: Hangup
    Get Signal: Interrupt


Block & Unblock signal
----------------------

.. code-block:: c

    #include <stdio.h>
    #include <string.h>
    #include <errno.h>
    #include <unistd.h>
    #include <signal.h>
    #include <setjmp.h>

    static sigjmp_buf jmpbuf;

    void handler(int signo)
    {
        printf("Get signal[%d]: %s\n", signo, sys_siglist[signo]);
        if (SIGUSR1 == signo) {
            siglongjmp(jmpbuf, 1);
        }
    }

    int main(int argc, char *argv[])
    {
        int ret = -1;
        sigset_t new_mask, old_mask;

        sigemptyset(&new_mask);
        sigaddset(&new_mask, SIGHUP);

        if (SIG_ERR == signal(SIGHUP, handler)) {
                printf("Set signal get %s error", strerror(errno));
                goto Error;
        }
        if (SIG_ERR == signal(SIGALRM, handler)) {
                printf("Set signal get %s error", strerror(errno));
                goto Error;
        }
        if (SIG_ERR == signal(SIGUSR1, handler)) {
                printf("Set signal get %s error", strerror(errno));
                goto Error;
        }
        /* block SIGHUP */
        if (sigsetjmp(jmpbuf, 1)) {
                /* unblock SIGHUP */
                sigprocmask(SIG_UNBLOCK, &new_mask, &old_mask);
        } else {
                /* block SIGHUP */
                sigprocmask(SIG_BLOCK, &new_mask, &old_mask);
        }
        while (1) sleep(3);
        ret = 0;
    Error:
        return ret;
    }

output:

.. code-block:: console

    $ kill -HUP %1
    $ kill -ALRM %1
    Get signal[14]: Alarm clock
    $ kill -USR1 %1
    Get signal[10]: User defined signal 1
    Get signal[1]: Hangup
