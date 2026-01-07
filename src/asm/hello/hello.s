# Hello World using 64-bit syscall
# Build: gcc -nostdlib -no-pie -o hello hello.s
# Run: ./hello

.global _start
.section .text

_start:
    # write(1, message, 13)
    mov $1, %rax        # syscall: write
    mov $1, %rdi        # fd: stdout
    lea message(%rip), %rsi  # buf: message (RIP-relative)
    mov $13, %rdx       # count
    syscall

    # exit(0)
    mov $60, %rax       # syscall: exit
    xor %rdi, %rdi      # status: 0
    syscall

.section .rodata
message:
    .ascii "Hello World\n"
