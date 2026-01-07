add_executable(procedure_asm_test procedure.s)
target_link_options(procedure_asm_test PRIVATE -nostdlib -no-pie)
add_test(NAME Asm.Procedure COMMAND procedure_asm_test)
