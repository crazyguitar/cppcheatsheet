add_executable(loop_asm_test loop.s)
target_link_options(loop_asm_test PRIVATE -nostdlib -no-pie)
add_test(NAME Asm.Loop COMMAND loop_asm_test)
