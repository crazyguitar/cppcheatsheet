# Function call example with stack frame
# Build: gcc -nostdlib -no-pie -o procedure procedure.s
# Run: ./procedure

.global _start
.section .text

_start:
    call print_hello
    call print_hello

    # exit(0)
    mov $60, %rax
    xor %rdi, %rdi
    syscall

# void print_hello(void)
print_hello:
    push %rbp           # save base pointer
    mov %rsp, %rbp      # set up stack frame

    mov $1, %rax        # write syscall
    mov $1, %rdi        # stdout
    lea message(%rip), %rsi
    mov $13, %rdx
    syscall

    pop %rbp            # restore base pointer
    ret

.section .rodata
message:
    .ascii "Hello World\n"
