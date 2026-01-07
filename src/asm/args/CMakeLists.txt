add_executable(args_asm_test args.s)
target_link_options(args_asm_test PRIVATE -nostdlib -no-pie)
add_test(NAME Asm.Args COMMAND args_asm_test foo bar baz)
