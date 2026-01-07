// Syscall wrapper examples (cross-platform)
// Build: gcc -O2 -o syscall-wrapper main.c
// Run: ./syscall-wrapper

#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/types.h>

#if defined(__x86_64__)

// x86_64 Linux syscall numbers
#define SYS_WRITE   1
#define SYS_GETPID  39
#define SYS_GETUID  102

static inline long syscall0(long n) {
    long ret;
    __asm__ volatile ("syscall" : "=a"(ret) : "a"(n) : "rcx", "r11", "memory");
    return ret;
}

static inline long syscall3(long n, long a1, long a2, long a3) {
    long ret;
    __asm__ volatile ("syscall" : "=a"(ret)
        : "a"(n), "D"(a1), "S"(a2), "d"(a3) : "rcx", "r11", "memory");
    return ret;
}

#elif defined(__aarch64__)

// ARM64 Linux syscall numbers
#define SYS_WRITE   64
#define SYS_GETPID  172
#define SYS_GETUID  174

static inline long syscall0(long n) {
    register long x8 __asm__("x8") = n;
    register long x0 __asm__("x0");
    __asm__ volatile ("svc #0" : "=r"(x0) : "r"(x8) : "memory");
    return x0;
}

static inline long syscall3(long n, long a1, long a2, long a3) {
    register long x8 __asm__("x8") = n;
    register long x0 __asm__("x0") = a1;
    register long x1 __asm__("x1") = a2;
    register long x2 __asm__("x2") = a3;
    __asm__ volatile ("svc #0" : "+r"(x0) : "r"(x8), "r"(x1), "r"(x2) : "memory");
    return x0;
}

#elif defined(__APPLE__)

// macOS uses different syscall mechanism, use libc
#define SYS_WRITE   0
#define SYS_GETPID  0
#define SYS_GETUID  0

static inline long syscall0(long n) { (void)n; return -1; }
static inline long syscall3(long n, long a1, long a2, long a3) {
    (void)n; (void)a1; (void)a2; (void)a3; return -1;
}

#endif

int main(void) {
#if defined(__x86_64__) && !defined(__APPLE__)
    printf("Architecture: x86_64 Linux\n\n");

    const char msg[] = "Hello from syscall wrapper!\n";
    syscall3(SYS_WRITE, 1, (long)msg, sizeof(msg) - 1);

    printf("getpid() via syscall: %ld\n", syscall0(SYS_GETPID));
    printf("getpid() via libc:    %d\n", getpid());
    printf("getuid() via syscall: %ld\n", syscall0(SYS_GETUID));
    printf("getuid() via libc:    %d\n", getuid());

#elif defined(__aarch64__) && !defined(__APPLE__)
    printf("Architecture: ARM64 Linux\n\n");

    const char msg[] = "Hello from syscall wrapper!\n";
    syscall3(SYS_WRITE, 1, (long)msg, sizeof(msg) - 1);

    printf("getpid() via syscall: %ld\n", syscall0(SYS_GETPID));
    printf("getpid() via libc:    %d\n", getpid());
    printf("getuid() via syscall: %ld\n", syscall0(SYS_GETUID));
    printf("getuid() via libc:    %d\n", getuid());

#else
    // macOS or other platforms - just use libc
    printf("Platform: macOS or other (using libc)\n\n");
    printf("getpid(): %d\n", getpid());
    printf("getuid(): %d\n", getuid());
#endif

    return 0;
}
