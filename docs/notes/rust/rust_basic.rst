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

See :doc:`rust_container` for details on slices (``&[T]``), ``Vec<T>``, and
fixed-size arrays (``[T; N]``).

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

Struct Mutability
-----------------

In Rust, mutability is a property of the **binding**, not individual fields. You cannot
make some fields mutable and others immutable — the entire struct is either mutable or
immutable based on how the variable is declared.

**C++ (per-field mutability):**

.. code-block:: cpp

    struct Foo {
        int x;                 // mutable
        const int y;           // immutable per-field
        mutable int z;         // always mutable, even on const instance
    };

    const Foo foo{1, 2, 3};
    // foo.x = 10;        // error: foo is const
    foo.z = 42;           // OK: mutable member

**Rust (whole-struct mutability):**

.. code-block:: rust

    struct Foo {
        x: i32,
        y: i32,
        z: i32,
    }

    let foo = Foo { x: 1, y: 2, z: 3 };  // immutable - ALL fields
    // foo.x = 10;                         // error: cannot mutate

    let mut foo = Foo { x: 1, y: 2, z: 3 };  // mutable - ALL fields
    foo.x = 10;                                // OK
    foo.y = 20;                                // OK

Nested Struct Mutability
~~~~~~~~~~~~~~~~~~~~~~~~

When a struct contains another struct, the nested struct inherits the mutability of
the parent binding. If the parent is ``mut``, all nested fields are mutable too.

.. code-block:: rust

    struct Bar {
        val: i32,
    }

    struct Foo {
        bar: Bar,
    }

    let mut foo = Foo { bar: Bar { val: 1 } };
    foo.bar.val = 10;          // OK - parent is mut, so nested fields are mut

    let foo2 = Foo { bar: Bar { val: 1 } };
    // foo2.bar.val = 10;      // error: entire tree is immutable

Interior Mutability
~~~~~~~~~~~~~~~~~~~

When you need to mutate data behind an immutable reference, Rust provides **interior
mutability** types that move the borrow check to runtime:

.. code-block:: rust

    use std::cell::RefCell;
    use std::rc::Rc;

    struct Foo {
        val: Rc<RefCell<i32>>,   // shared ownership + interior mutability
    }

    let foo = Foo { val: Rc::new(RefCell::new(1)) };
    *foo.val.borrow_mut() = 42;  // mutate through immutable binding

For thread-safe interior mutability, use ``Arc<Mutex<T>>``:

.. code-block:: rust

    use std::sync::{Arc, Mutex};

    let data = Arc::new(Mutex::new(0));
    let data2 = Arc::clone(&data);

    // Mutate from any thread holding a clone
    *data2.lock().unwrap() = 42;

Summary table:

.. list-table::
   :header-rows: 1

   * - Pattern
     - Ownership
     - Thread-safe
     - Check
   * - ``&mut T``
     - single owner
     - N/A
     - compile-time
   * - ``RefCell<T>``
     - single owner
     - No
     - runtime
   * - ``Rc<RefCell<T>>``
     - multiple owners
     - No
     - runtime
   * - ``Arc<Mutex<T>>``
     - multiple owners
     - Yes
     - runtime

Pointers and References
-----------------------

:Source: `src/rust/borrowing <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/borrowing>`_,
   `src/rust/lifetimes <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/lifetimes>`_

C++ has pointers (``T*``) and references (``T&``, ``T&&``). Rust splits these into
safe references (``&T``, ``&mut T``) and unsafe raw pointers (``*const T``, ``*mut T``).
The key difference: Rust references are always valid — the compiler guarantees they
never dangle.

.. list-table::
   :header-rows: 1

   * - C++
     - Rust
     - Notes
   * - ``const T&``
     - ``&T``
     - Shared (immutable) reference
   * - ``T&``
     - ``&mut T``
     - Exclusive (mutable) reference
   * - ``const T*``
     - ``*const T``
     - Raw pointer, requires ``unsafe``
   * - ``T*``
     - ``*mut T``
     - Raw mutable pointer, requires ``unsafe``
   * - ``T&&`` (rvalue ref)
     - *(no equivalent)*
     - Rust uses move semantics by default

**C++:**

.. code-block:: cpp

    void increment(int& x) { x += 1; }       // mutable ref
    void print(const int& x) { cout << x; }  // immutable ref

    struct Config {
      const std::string& name;  // reference member — no lifetime check!
      int value;

      void print() const {
        std::cout << name << "=" << value << "\n";
      }
    };

    int main() {
      int val = 5;
      increment(val);
      print(val);

      // BUG: reference member can easily dangle
      Config* c;
      {
        std::string s = "timeout";
        c = new Config{s, 30};
      } // s destroyed — c->name is now dangling!
      c->print();  // undefined behavior
    }

**Rust:**

.. code-block:: rust

    fn increment(x: &mut i32) { *x += 1; }   // exclusive reference
    fn print_val(x: &i32) { println!("{x}"); } // shared reference

    // Struct holding a reference MUST declare a lifetime parameter.
    // This tells the compiler: Config cannot outlive the data it borrows.
    struct Config<'a> {
        name: &'a str,
        value: i32,
    }

    impl<'a> Config<'a> {
        fn new(name: &'a str, value: i32) -> Self {
            Config { name, value }
        }

        // Accessing members — return lifetime tied to struct's lifetime
        fn name(&self) -> &str { self.name }

        fn display(&self) {
            println!("{}={}", self.name, self.value);
        }
    }

    fn main() {
        let mut val = 5;
        increment(&mut val);
        print_val(&val);

        // Compiler ensures Config cannot outlive the borrowed string
        let name = String::from("timeout");
        let cfg = Config::new(&name, 30);
        cfg.display();  // timeout=30

        // This would NOT compile — Rust prevents the dangling reference:
        // let cfg;
        // {
        //     let name = String::from("timeout");
        //     cfg = Config::new(&name, 30);
        // } // name dropped here
        // cfg.display();  // error: `name` does not live long enough
    }

Borrowing rules (enforced at compile time):

- You can have **many** ``&T`` OR **one** ``&mut T`` — never both at the same time
- References must always be valid (no dangling)

Dereferencing and Auto-deref
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For primitives, you must explicitly dereference with ``*``. But for struct member
access, Rust's ``.`` operator auto-dereferences — no ``->`` operator like C++:

**C++:**

.. code-block:: cpp

    struct Point { int x, y; };

    Point p{1, 2};
    Point& r = p;
    Point* ptr = &p;

    r.x;      // dot for references
    ptr->x;   // arrow for pointers
    (*ptr).x; // or explicit deref + dot

**Rust:**

.. code-block:: rust

    struct Point { x: i32, y: i32 }

    let p = Point { x: 1, y: 2 };
    let r = &p;
    let b = Box::new(Point { x: 3, y: 4 });

    r.x;       // auto-deref: same as (*r).x
    b.x;       // auto-deref through Box too
    // No -> operator in Rust — dot handles everything

    // Explicit * only needed for primitives
    let mut val = 5;
    let m = &mut val;
    *m += 1;   // must deref to assign to i32

Lifetimes
~~~~~~~~~

When a struct holds a reference or a function returns one, Rust needs to know how
long it's valid. In C++, this is entirely the programmer's responsibility (and a
common source of bugs). Rust makes it explicit with lifetime annotations ``'a``.

**C++ (dangling reference — compiles, crashes at runtime):**

.. code-block:: cpp

    struct Parser {
      const std::string& input;  // no lifetime tracking

      // Can easily outlive the string it references
      std::string_view next_token() const {
        return std::string_view(input).substr(0, input.find(' '));
      }
    };

**Rust (lifetime annotation — compiler enforces validity):**

.. code-block:: rust

    struct Parser<'a> {
        input: &'a str,  // 'a = "input must live at least as long as Parser"
    }

    impl<'a> Parser<'a> {
        fn new(input: &'a str) -> Self {
            Parser { input }
        }

        // Return type borrows from self, which borrows from 'a
        fn next_token(&self) -> &str {
            self.input.split_whitespace().next().unwrap_or("")
        }
    }

    let text = String::from("hello world");
    let parser = Parser::new(&text);
    println!("{}", parser.next_token());  // "hello"

    // Won't compile — parser cannot outlive text:
    // let parser;
    // {
    //     let text = String::from("hello world");
    //     parser = Parser::new(&text);
    // }
    // parser.next_token();  // error: `text` does not live long enough

Multiple lifetimes — when a struct borrows from different sources:

.. code-block:: rust

    // 'a and 'b can be different lifetimes
    struct Pair<'a, 'b> {
        key: &'a str,
        value: &'b str,
    }

    impl<'a, 'b> Pair<'a, 'b> {
        fn key(&self) -> &'a str { self.key }
        fn value(&self) -> &'b str { self.value }
    }

When you don't need lifetime annotations (lifetime elision rules):

.. code-block:: rust

    // Compiler infers: input and output have the same lifetime
    fn first_word(s: &str) -> &str {
        s.split_whitespace().next().unwrap_or("")
    }

    // Methods on &self — return lifetime tied to self automatically
    impl<'a> Parser<'a> {
        fn peek(&self) -> &str {  // no annotation needed
            self.input
        }
    }

``'static`` is a special lifetime meaning "valid for the entire program":

.. code-block:: rust

    // String literals are always &'static str
    let s: &'static str = "I live forever";

    // Struct can hold 'static references without lifetime parameter issues
    let cfg = Config::new("timeout", 30);  // &'static str, always valid

See Also
--------

- :doc:`rust_ownership` - Ownership, borrowing rules, and borrow checker
- :doc:`rust_raii` - Resource management and Drop trait
- :doc:`rust_smartptr` - Box, Rc, Arc, RefCell
