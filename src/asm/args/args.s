# Command line arguments example
# Build: gcc -nostdlib -no-pie -o args args.s
# Run: ./args foo bar baz

.global _start
.section .text

_start:
    # On entry: (%rsp) = argc, 8(%rsp) = argv[0], 16(%rsp) = argv[1], ...
    mov (%rsp), %r12        # r12 = argc
    lea 8(%rsp), %r13       # r13 = &argv[0]

.print_args:
    cmp $0, %r12
    je .done

    # Get string length
    mov (%r13), %rdi        # current argv
    call strlen             # length in %rax

    # write(1, argv[i], len)
    mov %rax, %rdx          # count
    mov $1, %rax            # write
    mov $1, %rdi            # stdout
    mov (%r13), %rsi        # buf
    syscall

    # write newline
    mov $1, %rax
    mov $1, %rdi
    lea newline(%rip), %rsi
    mov $1, %rdx
    syscall

    add $8, %r13            # next argv
    dec %r12                # argc--
    jmp .print_args

.done:
    mov $60, %rax
    xor %rdi, %rdi
    syscall

# size_t strlen(const char *s) - returns length in %rax
strlen:
    xor %rax, %rax          # len = 0
.strlen_loop:
    cmpb $0, (%rdi, %rax)   # if (s[len] == 0)
    je .strlen_done
    inc %rax                # len++
    jmp .strlen_loop
.strlen_done:
    ret

.section .rodata
newline:
    .ascii "\n"
