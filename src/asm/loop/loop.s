# Loop example: print message 5 times
# Build: gcc -nostdlib -no-pie -o loop loop.s
# Run: ./loop

.global _start
.section .text

_start:
    mov $5, %r12        # counter = 5

.loop:
    # write(1, message, 13)
    mov $1, %rax
    mov $1, %rdi
    lea message(%rip), %rsi
    mov $13, %rdx
    syscall

    dec %r12            # counter--
    jnz .loop           # if (counter != 0) goto loop

    # exit(0)
    mov $60, %rax
    xor %rdi, %rdi
    syscall

.section .rodata
message:
    .ascii "Hello World\n"
