add_executable(syscall_wrapper_test main.c)
add_test(NAME Asm.Syscall_wrapper COMMAND syscall_wrapper_test)
