// CPUID / CPU feature detection (cross-platform)
// Build: gcc -O2 -o cpuid main.c
// Run: ./cpuid

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#if defined(__x86_64__) || defined(__i386__)

typedef struct { uint32_t eax, ebx, ecx, edx; } cpuid_t;

static inline cpuid_t cpuid(uint32_t leaf, uint32_t subleaf) {
    cpuid_t r;
    __asm__ volatile (
        "cpuid"
        : "=a"(r.eax), "=b"(r.ebx), "=c"(r.ecx), "=d"(r.edx)
        : "a"(leaf), "c"(subleaf)
    );
    return r;
}

void print_cpu_info(void) {
    char vendor[13] = {0}, brand[49] = {0};
    cpuid_t id = cpuid(0, 0);

    memcpy(vendor + 0, &id.ebx, 4);
    memcpy(vendor + 4, &id.edx, 4);
    memcpy(vendor + 8, &id.ecx, 4);
    printf("Vendor: %s\n", vendor);

    for (int i = 0; i < 3; i++) {
        id = cpuid(0x80000002 + i, 0);
        memcpy(brand + i*16 + 0, &id.eax, 4);
        memcpy(brand + i*16 + 4, &id.ebx, 4);
        memcpy(brand + i*16 + 8, &id.ecx, 4);
        memcpy(brand + i*16 + 12, &id.edx, 4);
    }
    printf("Brand:  %s\n", brand);

    cpuid_t f = cpuid(1, 0);
    printf("\nFeatures:\n");
    printf("  SSE3:   %s\n", (f.ecx & (1 << 0))  ? "yes" : "no");
    printf("  SSE4.1: %s\n", (f.ecx & (1 << 19)) ? "yes" : "no");
    printf("  SSE4.2: %s\n", (f.ecx & (1 << 20)) ? "yes" : "no");
    printf("  AVX:    %s\n", (f.ecx & (1 << 28)) ? "yes" : "no");
    printf("  AES-NI: %s\n", (f.ecx & (1 << 25)) ? "yes" : "no");

    cpuid_t e = cpuid(7, 0);
    printf("  AVX2:   %s\n", (e.ebx & (1 << 5))  ? "yes" : "no");
}

#elif defined(__aarch64__) && defined(__linux__)

void print_cpu_info(void) {
    uint64_t isar0;
    __asm__ volatile ("mrs %0, id_aa64isar0_el1" : "=r"(isar0));

    printf("ID_AA64ISAR0_EL1: 0x%016llx\n", (unsigned long long)isar0);

    printf("\nFeatures:\n");
    printf("  AES:    %s\n", ((isar0 >> 4) & 0xF)  ? "yes" : "no");
    printf("  SHA1:   %s\n", ((isar0 >> 8) & 0xF)  ? "yes" : "no");
    printf("  SHA256: %s\n", ((isar0 >> 12) & 0xF) ? "yes" : "no");
    printf("  CRC32:  %s\n", ((isar0 >> 16) & 0xF) ? "yes" : "no");
    printf("  Atomic: %s\n", ((isar0 >> 20) & 0xF) ? "yes" : "no");
}

#elif defined(__APPLE__) && defined(__aarch64__)

#include <sys/sysctl.h>

void print_cpu_info(void) {
    char brand[256] = {0};
    size_t len = sizeof(brand);
    sysctlbyname("machdep.cpu.brand_string", brand, &len, NULL, 0);
    printf("Brand: %s\n", brand);

    int64_t features = 0;
    len = sizeof(features);

    printf("\nFeatures:\n");
    int has_aes = 0, has_sha1 = 0, has_sha256 = 0;
    len = sizeof(has_aes);
    sysctlbyname("hw.optional.arm.FEAT_AES", &has_aes, &len, NULL, 0);
    sysctlbyname("hw.optional.arm.FEAT_SHA1", &has_sha1, &len, NULL, 0);
    sysctlbyname("hw.optional.arm.FEAT_SHA256", &has_sha256, &len, NULL, 0);

    printf("  AES:    %s\n", has_aes ? "yes" : "no");
    printf("  SHA1:   %s\n", has_sha1 ? "yes" : "no");
    printf("  SHA256: %s\n", has_sha256 ? "yes" : "no");
}

#else

void print_cpu_info(void) {
    printf("CPU info not available on this platform\n");
}

#endif

int main(void) {
#if defined(__x86_64__)
    printf("Architecture: x86_64\n\n");
#elif defined(__aarch64__)
    printf("Architecture: ARM64\n\n");
#endif
    print_cpu_info();
    return 0;
}
