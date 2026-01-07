# Exit using 64-bit syscall
# Build: gcc -nostdlib -no-pie -o exit64 exit64.s
# Run: ./exit64; echo $?

.global _start
.section .text

_start:
    mov $60, %rax       # syscall: exit (64-bit)
    mov $42, %rdi       # exit code
    syscall             # invoke 64-bit syscall

.section .data
