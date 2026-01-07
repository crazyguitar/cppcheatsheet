// GCC inline assembly examples (cross-platform)
// Build: gcc -O2 -o inline-asm main.c
// Run: ./inline-asm

#include <stdio.h>
#include <stdint.h>

#if defined(__x86_64__) || defined(__i386__)
// x86/x86_64 specific

static inline uint64_t rdtsc(void) {
    uint32_t lo, hi;
    __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

static inline uint32_t bswap32(uint32_t x) {
    __asm__ ("bswapl %0" : "+r"(x));
    return x;
}

#elif defined(__aarch64__)
// ARM64 specific

static inline uint64_t rdtsc(void) {
    uint64_t val;
    __asm__ volatile ("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline uint32_t bswap32(uint32_t x) {
    __asm__ ("rev %w0, %w0" : "+r"(x));
    return x;
}

#endif

// Cross-platform: simple add using inline asm
static inline int add_asm(int a, int b) {
    int result;
#if defined(__x86_64__) || defined(__i386__)
    __asm__ ("addl %2, %0" : "=r"(result) : "0"(a), "r"(b));
#elif defined(__aarch64__)
    __asm__ ("add %w0, %w1, %w2" : "=r"(result) : "r"(a), "r"(b));
#else
    result = a + b;
#endif
    return result;
}

// Cross-platform: memory barrier
static inline void memory_barrier(void) {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ volatile ("mfence" ::: "memory");
#elif defined(__aarch64__)
    __asm__ volatile ("dmb ish" ::: "memory");
#else
    __sync_synchronize();
#endif
}

// Cross-platform: compiler barrier only
static inline void compiler_barrier(void) {
    __asm__ volatile ("" ::: "memory");
}

int main(void) {
#if defined(__x86_64__)
    printf("Architecture: x86_64\n\n");
#elif defined(__i386__)
    printf("Architecture: i386\n\n");
#elif defined(__aarch64__)
    printf("Architecture: ARM64\n\n");
#else
    printf("Architecture: unknown\n\n");
#endif

    // rdtsc / cycle counter
    uint64_t t1 = rdtsc();
    uint64_t t2 = rdtsc();
    printf("Cycle counter: %llu cycles between calls\n",
           (unsigned long long)(t2 - t1));

    // add
    printf("add_asm(10, 32) = %d\n", add_asm(10, 32));

    // bswap
    printf("bswap32(0x12345678) = 0x%08x\n", bswap32(0x12345678));

    return 0;
}
