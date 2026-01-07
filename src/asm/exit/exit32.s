# Exit using 32-bit syscall (int 0x80)
# Build: gcc -nostdlib -no-pie -o exit32 exit32.s
# Run: ./exit32; echo $?

.global _start
.section .text

_start:
    mov $1, %eax        # syscall: exit (32-bit)
    mov $42, %ebx       # exit code
    int $0x80           # invoke 32-bit syscall

.section .data
