================
Rust Programming
================

.. meta::
   :description: Rust programming guide for C++ developers. Covers ownership, borrowing, structs, traits, generics, error handling, smart pointers, concurrency, and modules with C++ comparisons.
   :keywords: Rust, C++, learn Rust, ownership, borrowing, traits, generics, Result, Option, smart pointers, Arc, Mutex, modules, cargo

Rust is a systems programming language that guarantees memory safety without garbage
collection. For C++ developers, Rust offers familiar concepts like RAII, zero-cost
abstractions, and low-level control, while adding compile-time safety guarantees
that prevent data races, null pointer dereferences, and use-after-free bugs.

This section maps C++ concepts to their Rust equivalents, highlighting similarities
and key differences. Each chapter includes side-by-side code comparisons with
working examples.

.. toctree::
   :maxdepth: 1

   rust_basic
   rust_raii
   rust_string
   rust_container
   rust_iterator
   rust_traits
   rust_casting
   rust_constfn
   rust_closure
   rust_smartptr
   rust_error
   rust_thread
   rust_modules
