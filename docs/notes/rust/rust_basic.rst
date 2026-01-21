===========
Rust Basics
===========

.. meta::
   :description: Rust basics for C++ developers covering variables, mutability, ownership, borrowing, and references with side-by-side C++ comparisons.
   :keywords: Rust, C++, variables, mutability, ownership, borrowing, references, move semantics, let, mut

.. contents:: Table of Contents
    :backlinks: none

This chapter covers fundamental Rust concepts that differ most from C++: immutability
by default, ownership, and borrowing. Understanding these concepts is essential for
writing idiomatic Rust code.

Variables and Mutability
------------------------

:Source: `src/rust/variables <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/variables>`_

In C++, variables are mutable by default and you use ``const`` to make them immutable.
Rust takes the opposite approach: variables are immutable by default, and you must
explicitly use the ``mut`` keyword to allow mutation.

**C++:**

.. code-block:: cpp

    int x = 5;           // mutable by default
    x = 10;              // OK
    const int y = 5;     // immutable
    // y = 10;           // error

**Rust:**

.. code-block:: rust

    let x = 5;           // immutable by default
    // x = 10;           // error: cannot assign twice to immutable variable
    let mut y = 5;       // mutable
    y = 10;              // OK

Shadowing
~~~~~~~~~

Rust allows re-declaring a variable with the same name, which shadows the previous
binding. This is different from mutation and allows changing the type.

**C++ does not support shadowing in the same scope.** Re-declaring a variable with
the same name is a compilation error. You must use a different variable name or
explicitly cast/convert the value.

**C++ (not allowed):**

.. code-block:: cpp

    int x = 5;
    int x = x + 1;       // error: redefinition of 'x'
    // const char* x = "hello";  // error: redefinition with different type

    // Workaround: use different names
    int x1 = 5;
    int x2 = x1 + 1;
    const char* x3 = "hello";

**Rust (shadowing allowed):**

.. code-block:: rust

    let x = 5;
    let x = x + 1;       // shadows previous x, x is now 6
    let x = "hello";     // shadows again, different type - OK in Rust

Shadowing is useful in Rust for transforming a value while keeping the same name,
especially when parsing or converting types:

Ownership
---------

:Source: `src/rust/ownership <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/ownership>`_

C++ has copy semantics by default. Move semantics were added in C++11 via ``std::move``,
but using a moved-from object is undefined behavior that the compiler won't catch.

Rust uses move semantics by default for types that manage resources. When you assign
a ``String`` to another variable, ownership transfers and the original becomes invalid.
The compiler enforces this, preventing use-after-move bugs.

**C++:**

.. code-block:: cpp

    #include <string>
    #include <utility>

    int main() {
      std::string s1 = "hello";
      std::string s2 = s1;              // copy (deep clone)
      std::string s3 = std::move(s1);   // move, s1 is now in valid but unspecified state
      // Using s1 here is undefined behavior but compiles
    }

**Rust:**

.. code-block:: rust

    fn main() {
        let s1 = String::from("hello");
        let s2 = s1.clone();  // explicit clone (deep copy)
        let s3 = s1;          // move, s1 is no longer valid
        // println!("{}", s1); // error: borrow of moved value
    }

Copy vs Move Types
~~~~~~~~~~~~~~~~~~

Types that implement the ``Copy`` trait (like integers, floats, bools) are copied
implicitly. Types that manage heap resources (like ``String``, ``Vec``) are moved.

**In C++, all types are copyable by default** (unless explicitly deleted). There's no
built-in distinction between "copy types" and "move types" - the programmer must
remember which types are expensive to copy.

**C++:**

.. code-block:: cpp

    int x = 5;
    int y = x;      // copy (cheap, primitive)

    std::string s1 = "hello";
    std::string s2 = s1;    // copy (expensive, heap allocation)
    // Both s1 and s2 are valid - no compiler help

**Rust:**

.. code-block:: rust

    // Copy types - implicit copy
    let x = 5;
    let y = x;      // copy, both x and y are valid
    println!("{} {}", x, y);  // OK

    // Move types - ownership transfer
    let s1 = String::from("hello");
    let s2 = s1;    // move, s1 is invalid
    // println!("{}", s1);  // error: use of moved value

References and Borrowing
------------------------

:Source: `src/rust/borrowing <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/borrowing>`_

C++ references are aliases that the programmer must ensure don't outlive their data.
Rust's borrow checker enforces at compile time that:

1. You can have either one mutable reference OR any number of immutable references
2. References must always be valid (no dangling references)

**C++:**

.. code-block:: cpp

    void modify(int& x) { x += 1; }
    void read(const int& x) { std::cout << x; }

    int main() {
      int val = 5;
      modify(val);     // mutable reference
      read(val);       // const reference
      // No compile-time check for dangling references
    }

**Rust:**

.. code-block:: rust

    fn modify(x: &mut i32) { *x += 1; }
    fn read(x: &i32) { println!("{}", x); }

    fn main() {
        let mut val = 5;
        modify(&mut val);  // mutable borrow
        read(&val);        // immutable borrow
        // Compiler guarantees no dangling references
    }

Borrowing Rules
~~~~~~~~~~~~~~~

**C++ has no equivalent compile-time enforcement.** You can have multiple pointers
or references to the same data, with any combination of const/non-const, and the
compiler won't prevent data races or aliasing bugs.

**C++ (compiles but dangerous):**

.. code-block:: cpp

    std::string s = "hello";
    std::string& r1 = s;        // mutable reference
    const std::string& r2 = s;  // const reference
    r1 += " world";             // modifies s while r2 exists
    // No compiler error, but can cause subtle bugs in multithreaded code

**Rust (enforced at compile time):**

.. code-block:: rust

    let mut s = String::from("hello");

    // Multiple immutable borrows - OK
    let r1 = &s;
    let r2 = &s;
    println!("{} {}", r1, r2);

    // Mutable borrow after immutable borrows end - OK
    let r3 = &mut s;
    r3.push_str(" world");

    // Cannot have mutable and immutable borrows simultaneously
    // let r4 = &s;
    // let r5 = &mut s;  // error: cannot borrow as mutable

Lifetimes
---------

Lifetimes are Rust's way of ensuring references don't outlive the data they point to.
Most of the time, lifetimes are inferred. When the compiler can't infer them, you
must annotate explicitly.

**C++ (dangling reference - compiles but UB):**

.. code-block:: cpp

    int& get_ref() {
      int x = 5;
      return x;  // dangling reference - undefined behavior
    }

**Rust (compile error):**

.. code-block:: rust

    fn get_ref() -> &i32 {
        let x = 5;
        &x  // error: cannot return reference to local variable
    }

Explicit Lifetime Annotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a function takes multiple references and returns a reference, you may need
to specify how the lifetimes relate.

**C++ has no equivalent syntax.** The programmer must document and manually ensure
that returned references remain valid. The compiler provides no help.

**C++ (no lifetime tracking):**

.. code-block:: cpp

    // Programmer must ensure returned reference outlives usage
    // No way to express "return value lives as long as inputs"
    const std::string& longest(const std::string& x, const std::string& y) {
        return x.length() > y.length() ? x : y;
    }

    // Dangerous: easy to return reference to temporary
    const std::string& dangerous(const std::string& x) {
        std::string temp = x + "!";
        return temp;  // dangling reference - compiles but UB
    }

**Rust (explicit lifetime annotations):**

.. code-block:: rust

    // 'a is a lifetime parameter - return value lives as long as inputs
    fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
        if x.len() > y.len() { x } else { y }
    }

    fn main() {
        let s1 = String::from("long string");
        let s2 = String::from("short");
        let result = longest(&s1, &s2);
        println!("Longest: {}", result);
    }

Slices
------

Slices are references to a contiguous sequence of elements. They're similar to
C++20's ``std::span`` but are a fundamental part of Rust.

**C++ (std::span, C++20):**

.. code-block:: cpp

    #include <span>
    #include <vector>

    void print_slice(std::span<int> s) {
      for (int x : s) std::cout << x << " ";
    }

    int main() {
      std::vector<int> v = {1, 2, 3, 4, 5};
      print_slice(std::span(v).subspan(1, 3));  // 2 3 4
    }

**Rust:**

.. code-block:: rust

    fn print_slice(s: &[i32]) {
        for x in s {
            print!("{} ", x);
        }
    }

    fn main() {
        let v = vec![1, 2, 3, 4, 5];
        print_slice(&v[1..4]);  // 2 3 4
    }

See Also
--------

- :doc:`rust_raii` - Resource management and Drop trait
- :doc:`rust_smartptr` - Box, Rc, Arc, RefCell
