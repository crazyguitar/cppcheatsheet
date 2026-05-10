fn main() {
    // Compile C++ code and link it
    cc::Build::new()
        .cpp(true)
        .file("cpp-lib.cc")
        .compile("cpp_lib");
}
