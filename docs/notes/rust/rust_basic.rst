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

Underscore ``_`` Placeholder
----------------------------

:Source: `src/rust/underscore <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/underscore>`_

Rust uses ``_`` as a wildcard or placeholder in several contexts: type inference,
pattern matching, and ignoring unused values. C++ achieves similar goals with
``auto``, ``std::ignore``, and ``[[maybe_unused]]``, but Rust's ``_`` is more
versatile — especially for partial type inference, which C++ does not support.

Partial Type Inference
~~~~~~~~~~~~~~~~~~~~~~

Rust allows inferring individual type parameters using ``_``, while C++ ``auto``
is all-or-nothing — you either write the full type or let the compiler infer
everything.

**C++ (no partial inference):**

.. code-block:: cpp

    #include <vector>
    #include <map>

    // auto infers the entire type
    auto v = std::vector{1, 2, 3};             // deduces vector<int>
    auto m = std::map<std::string, int>{};     // must write full type or use auto

    // Cannot say: "I know it's a vector, infer the element type"
    // std::vector<auto> v = {1, 2, 3};        // not valid C++

**Rust (partial inference with ``_``):**

.. code-block:: rust

    // Infer the element type, but specify the container
    let v: Vec<_> = vec![1, 2, 3];             // compiler infers Vec<i32>

    use std::collections::HashMap;
    let m: HashMap<_, _> = vec![
        ("key".to_string(), 1),
    ].into_iter().collect();                    // infers HashMap<String, i32>

    // Especially useful with collect() where the compiler
    // needs to know the target collection type
    let squares: Vec<_> = (0..5).map(|x| x * x).collect();

Ignoring Values in Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rust's ``_`` can discard values in destructuring and ``match`` expressions. C++
uses ``std::ignore`` with ``std::tie`` or unnamed variables in structured bindings.

**C++:**

.. code-block:: cpp

    #include <tuple>

    auto [x, _] = std::make_pair(1, 2);        // _ is just a variable name
    // Note: _ is not special in C++, it's a regular identifier

    int val;
    std::tie(val, std::ignore) = std::make_pair(1, 2);  // truly discards

**Rust:**

.. code-block:: rust

    // Destructuring — _ truly discards the value
    let (x, _) = (1, 2);

    // Match expressions
    let value = Some(42);
    match value {
        Some(_) => println!("has a value"),    // don't care what's inside
        None => println!("empty"),
    }

    // Ignoring parts of a struct
    struct Point { x: i32, y: i32, z: i32 }
    let p = Point { x: 1, y: 2, z: 3 };
    let Point { x, .. } = p;                   // ignore y and z

Suppressing Unused Warnings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**C++:**

.. code-block:: cpp

    [[maybe_unused]] int result = do_something();  // suppress warning (C++17)

**Rust:**

.. code-block:: rust

    let _result = do_something();   // prefix with _ to suppress warning

.. note::

    Rust's ``_`` is **not** like ``any`` in TypeScript or ``Object`` in Java.
    It does not mean "any type." The compiler still determines a single concrete
    type at compile time — ``_`` simply means "infer this type for me from context."

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

The ``?`` Operator
-------------------

Rust's ``?`` operator unwraps a ``Result`` on success or returns the error early.
It replaces verbose ``match`` blocks for error propagation.

**C++ (manual error checking):**

.. code-block:: cpp

    #include <fstream>
    #include <vector>
    #include <optional>

    std::optional<std::vector<char>> read_file(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            return std::nullopt;  // manual error check
        }
        std::vector<char> data((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
        return data;
    }

**Rust (with ``?`` operator):**

.. code-block:: rust

    use std::fs;
    use std::io;

    fn read_file(path: &str) -> io::Result<Vec<u8>> {
        let data = fs::read(path)?;  // returns Err early if file read fails
        Ok(data)
    }

The ``?`` operator is equivalent to:

.. code-block:: rust

    let data = match fs::read(path) {
        Ok(bytes) => bytes,        // unwrap the value
        Err(e) => return Err(e),   // propagate the error
    };

.. note::

    The ``?`` operator can only be used in functions that return ``Result``
    (or ``Option``). It does not work in ``main()`` unless ``main`` returns
    ``Result``.

``Result<T, E>`` and ``io::Result<T>``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Result`` is a generic enum with two type parameters:

.. code-block:: rust

    enum Result<T, E> {
        Ok(T),    // success — holds a value of type T
        Err(E),   // failure — holds an error of type E
    }

The standard library provides a type alias ``io::Result<T>`` for I/O operations,
which fixes the error type to ``io::Error``:

.. code-block:: rust

    // defined in std::io
    type Result<T> = Result<T, io::Error>;

So ``io::Result<()>`` expands to ``Result<(), io::Error>``:

- ``Ok(())`` — success, no return value (like ``void`` in C++)
- ``Err(io::Error)`` — failure, holds an I/O error

.. code-block:: rust

    use std::fs;
    use std::io;

    fn load(path: &str) -> io::Result<()> {
        let data = fs::read(path)?;  // could fail → io::Error
        println!("read {} bytes", data.len());
        Ok(())  // success, nothing to return
    }

See Also
--------

- :doc:`rust_raii` - Resource management and Drop trait
- :doc:`rust_smartptr` - Box, Rc, Arc, RefCell
