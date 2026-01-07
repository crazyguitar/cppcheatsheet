add_executable(cpuid_test main.c)
add_test(NAME Asm.Cpuid COMMAND cpuid_test)
