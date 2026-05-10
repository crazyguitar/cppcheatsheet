================================
Rust FFI - Calling C++ from Rust
================================

.. meta::
   :description: Complete guide to calling C++ libraries from Rust using FFI (Foreign Function Interface). Covers extern C linkage, raw pointers, CString/CStr for strings, repr(C) structs, safe wrappers, and bindgen for automatic binding generation.
   :keywords: Rust FFI, Rust C++ bindings, extern C Rust, call C++ from Rust, Rust unsafe, CString CStr, repr C, bindgen, cc crate, Rust interop, foreign function interface

.. contents:: Table of Contents
    :backlinks: none

Rust can call C and C++ code through its Foreign Function Interface (FFI). Since C++
has no stable ABI, C++ functions must be exposed with ``extern "C"`` linkage to be
callable from Rust. This chapter covers the fundamentals of binding C++ libraries,
handling different data types across the FFI boundary, and creating safe Rust wrappers.

The key challenge in FFI is that Rust's safety guarantees don't extend across the
boundary - all FFI calls are inherently ``unsafe``. The common pattern is to write
thin unsafe FFI declarations, then wrap them in safe Rust functions that enforce
proper usage.

Basic FFI Setup
---------------

:Source: `src/rust/ffi <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/ffi>`_

Setting up Rust to call C++ code requires three components working together: the C++
source code with C-compatible function signatures, a build script that compiles and
links the C++ code into your Rust project, and Rust declarations that tell the compiler
about the foreign functions. The ``cc`` crate handles the compilation complexity,
automatically detecting the system's C++ compiler and configuring the build correctly.

**Project structure:**

.. code-block:: text

    ffi/
    ├── Cargo.toml      # build-dependencies = { cc = "1.0" }
    ├── build.rs        # Compiles cpp-lib.cc
    ├── cpp-lib.cc      # C++ library with extern "C"
    └── main.rs         # Rust FFI declarations and wrappers

**Cargo.toml:**

The ``cc`` crate is specified as a build dependency since it's only needed during
compilation, not at runtime. This keeps your final binary free of unnecessary dependencies.

.. code-block:: toml

    [package]
    name = "ffi"
    version = "0.1.0"
    edition = "2021"

    [build-dependencies]
    cc = "1.0"

**build.rs:**

The build script runs before your Rust code compiles. It invokes the ``cc`` crate to
compile the C++ source file into a static library, which Cargo then automatically links
into your final executable. The ``.cpp(true)`` flag tells ``cc`` to use the C++ compiler
instead of the C compiler.

.. code-block:: rust

    fn main() {
        cc::Build::new()
            .cpp(true)           // Compile as C++
            .file("cpp-lib.cc")
            .compile("cpp_lib"); // Output library name
    }

Calling Simple Functions
------------------------

The simplest FFI case involves functions with primitive types like integers and floats.
These types have identical memory representations in both Rust and C++, so no conversion
is needed. The ``extern "C"`` block in Rust declares the function signature, and the
``unsafe`` block is required because the compiler cannot verify the C++ implementation
is correct.

**C++ (cpp-lib.cc):**

.. code-block:: cpp

    extern "C" {
        int32_t cpp_add(int32_t a, int32_t b) {
            return a + b;
        }
    }

**Rust:**

.. code-block:: rust

    extern "C" {
        fn cpp_add(a: i32, b: i32) -> i32;
    }

    // Safe wrapper
    pub fn add(a: i32, b: i32) -> i32 {
        unsafe { cpp_add(a, b) }
    }

    fn main() {
        println!("Result: {}", add(10, 20)); // 30
    }

**C++ equivalent (calling C from C++):**

.. code-block:: cpp

    // In C++, you'd use extern "C" to call C functions
    extern "C" int c_function(int x);

    int main() {
        int result = c_function(42);
    }

Passing Arrays and Pointers
---------------------------

When passing arrays across the FFI boundary, C and C++ represent them as raw pointers
with a separate length parameter. Rust's slice type (``&[T]`` or ``&mut [T]``) combines
the pointer and length into a single fat pointer, but this representation isn't compatible
with C. The safe wrapper pattern extracts the raw pointer and length from the slice,
passes them to the C++ function, and ensures the slice remains valid for the duration
of the call.

**C++:**

.. code-block:: cpp

    extern "C" {
        void cpp_fill_array(int32_t* arr, size_t len, int32_t value) {
            for (size_t i = 0; i < len; ++i) {
                arr[i] = value;
            }
        }
    }

**Rust:**

.. code-block:: rust

    extern "C" {
        fn cpp_fill_array(arr: *mut i32, len: usize, value: i32);
    }

    // Safe wrapper using slices
    pub fn fill_array(arr: &mut [i32], value: i32) {
        unsafe {
            cpp_fill_array(arr.as_mut_ptr(), arr.len(), value)
        }
    }

    fn main() {
        let mut arr = [0i32; 5];
        fill_array(&mut arr, 42);
        println!("{:?}", arr); // [42, 42, 42, 42, 42]
    }

**C++ equivalent:**

.. code-block:: cpp

    #include <span>  // C++20

    void fill_array(std::span<int32_t> arr, int32_t value) {
        for (auto& x : arr) x = value;
    }

String Handling
---------------

Strings are one of the trickiest types to pass across FFI boundaries because Rust and C
use fundamentally different string representations. Rust's ``String`` is a UTF-8 encoded,
length-prefixed, heap-allocated buffer without a null terminator. C strings are null-terminated
byte arrays with no length field. The ``CString`` type creates an owned, null-terminated
string suitable for passing to C, while ``CStr`` provides a borrowed view of a C string
for reading data returned from C functions.

**C++:**

.. code-block:: cpp

    extern "C" {
        // Returns heap-allocated string - caller must free
        char* cpp_create_greeting(const char* name) {
            const char* prefix = "Hello, ";
            const char* suffix = "!";
            size_t len = strlen(prefix) + strlen(name) + strlen(suffix) + 1;
            char* result = new char[len];
            strcpy(result, prefix);
            strcat(result, name);
            strcat(result, suffix);
            return result;
        }

        void cpp_free_string(char* s) {
            delete[] s;
        }
    }

**Rust:**

.. code-block:: rust

    use std::ffi::{CStr, CString};
    use std::os::raw::c_char;

    extern "C" {
        fn cpp_create_greeting(name: *const c_char) -> *mut c_char;
        fn cpp_free_string(s: *mut c_char);
    }

    pub fn create_greeting(name: &str) -> String {
        // Convert Rust &str to C string
        let c_name = CString::new(name).expect("CString::new failed");

        unsafe {
            let ptr = cpp_create_greeting(c_name.as_ptr());
            // Convert C string back to Rust String
            let result = CStr::from_ptr(ptr).to_string_lossy().into_owned();
            // Free the C++ allocated memory
            cpp_free_string(ptr);
            result
        }
    }

**Key types:**

- ``CString`` - Owned, null-terminated string for passing to C. Allocates memory and appends a null byte.
- ``CStr`` - Borrowed reference to a null-terminated string from C. Zero-cost wrapper around a ``*const c_char``.
- ``c_char`` - Platform-specific C char type (usually ``i8`` on most platforms).

Passing Structs
---------------

Structs can be shared between Rust and C++ when they have compatible memory layouts.
By default, Rust is free to reorder struct fields and add padding for optimization.
The ``#[repr(C)]`` attribute forces Rust to use the same field ordering and alignment
rules as C, making the struct binary-compatible. This is essential for any struct that
crosses the FFI boundary, whether passed by value or by pointer.

**C++:**

.. code-block:: cpp

    extern "C" {
        struct Point {
            double x;
            double y;
        };

        double cpp_distance(const Point* p1, const Point* p2) {
            double dx = p2->x - p1->x;
            double dy = p2->y - p1->y;
            return sqrt(dx * dx + dy * dy);
        }

        Point cpp_midpoint(const Point* p1, const Point* p2) {
            return Point{(p1->x + p2->x) / 2.0, (p1->y + p2->y) / 2.0};
        }
    }

**Rust:**

.. code-block:: rust

    #[repr(C)]  // Use C-compatible memory layout
    #[derive(Debug, Clone, Copy)]
    pub struct Point {
        pub x: f64,
        pub y: f64,
    }

    extern "C" {
        fn cpp_distance(p1: *const Point, p2: *const Point) -> f64;
        fn cpp_midpoint(p1: *const Point, p2: *const Point) -> Point;
    }

    pub fn distance(p1: &Point, p2: &Point) -> f64 {
        unsafe { cpp_distance(p1, p2) }
    }

    pub fn midpoint(p1: &Point, p2: &Point) -> Point {
        unsafe { cpp_midpoint(p1, p2) }
    }

    fn main() {
        let p1 = Point { x: 0.0, y: 0.0 };
        let p2 = Point { x: 3.0, y: 4.0 };
        println!("Distance: {}", distance(&p1, &p2));   // 5.0
        println!("Midpoint: {:?}", midpoint(&p1, &p2)); // Point { x: 1.5, y: 2.0 }
    }

Type Mapping Reference
----------------------

Common type mappings between Rust and C/C++:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Rust
     - C/C++
     - Notes
   * - ``i8``, ``i16``, ``i32``, ``i64``
     - ``int8_t``, ``int16_t``, ``int32_t``, ``int64_t``
     - Fixed-size integers
   * - ``u8``, ``u16``, ``u32``, ``u64``
     - ``uint8_t``, ``uint16_t``, ``uint32_t``, ``uint64_t``
     - Fixed-size unsigned
   * - ``f32``, ``f64``
     - ``float``, ``double``
     - Floating point
   * - ``bool``
     - ``bool``
     - Boolean (1 byte)
   * - ``usize``
     - ``size_t``
     - Pointer-sized unsigned
   * - ``isize``
     - ``ptrdiff_t``
     - Pointer-sized signed
   * - ``*const T``
     - ``const T*``
     - Immutable pointer
   * - ``*mut T``
     - ``T*``
     - Mutable pointer
   * - ``c_char``
     - ``char``
     - Platform-specific char
   * - ``c_void``
     - ``void``
     - Opaque type

Bindgen for Automatic Bindings
------------------------------

For large C/C++ libraries with hundreds of functions and types, manually writing FFI
declarations is tedious and error-prone. The ``bindgen`` tool solves this by parsing
C/C++ header files and automatically generating the corresponding Rust ``extern`` blocks,
struct definitions, and type aliases. This ensures your Rust declarations always match
the actual C++ signatures, eliminating a common source of subtle bugs.

**Cargo.toml:**

.. code-block:: toml

    [build-dependencies]
    bindgen = "0.69"
    cc = "1.0"

**build.rs with bindgen:**

The build script first compiles the C++ library, then runs bindgen to parse the header
file and generate Rust bindings. The generated code is written to the ``OUT_DIR``, a
Cargo-managed directory for build artifacts, and included in your Rust code at compile time.

.. code-block:: rust

    use std::env;
    use std::path::PathBuf;

    fn main() {
        // Compile C++ library
        cc::Build::new()
            .cpp(true)
            .file("cpp-lib.cc")
            .compile("cpp_lib");

        // Generate bindings from header
        let bindings = bindgen::Builder::default()
            .header("cpp_lib.h")
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .expect("Unable to generate bindings");

        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out_path.join("bindings.rs"))
            .expect("Couldn't write bindings");
    }

**Using generated bindings:**

.. code-block:: rust

    // Include the generated bindings
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

See Also
--------

- `The Rustonomicon - FFI <https://doc.rust-lang.org/nomicon/ffi.html>`_
- `bindgen User Guide <https://rust-lang.github.io/rust-bindgen/>`_
- `cc crate documentation <https://docs.rs/cc/latest/cc/>`_
