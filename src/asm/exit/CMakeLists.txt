add_executable(exit64_test exit64.s)
target_link_options(exit64_test PRIVATE -nostdlib -no-pie)
add_test(NAME Asm.Exit64 COMMAND sh -c "$<TARGET_FILE:exit64_test>; test $? -eq 42")
