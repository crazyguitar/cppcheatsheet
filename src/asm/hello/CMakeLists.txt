add_executable(hello_asm_test hello.s)
target_link_options(hello_asm_test PRIVATE -nostdlib -no-pie)
add_test(NAME Asm.Hello COMMAND hello_asm_test)
