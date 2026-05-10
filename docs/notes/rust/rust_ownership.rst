=========
Ownership
=========

.. meta::
   :description: Rust ownership, borrowing, and lifetimes explained for C++ developers. Covers move semantics, references vs borrowing, borrow checker, and lifetime annotations.
   :keywords: Rust, ownership, borrowing, lifetimes, borrow checker, references, move semantics, C++ comparison, dangling pointer, RAII

.. contents:: Table of Contents
    :backlinks: none

Rust's ownership system is its most distinctive feature, providing memory safety
without garbage collection. For C++ developers, ownership combines familiar concepts
(RAII, move semantics, references) with compile-time enforcement that prevents
dangling pointers, data races, and use-after-free bugs.

Ownership Rules
---------------

:Source: `src/rust/ownership <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/ownership>`_

Rust's three ownership rules:

1. Each value has exactly one owner
2. When the owner goes out of scope, the value is dropped
3. Ownership can be transferred (moved) or borrowed

In C++, there's no compiler enforcement of ownership. You can create pointers to
local variables, and the compiler won't stop you from using them after the pointed-to
object is destroyed. This leads to dangling pointer bugs that are notoriously difficult
to debug because they may appear to work in some runs but crash in others.

**C++ (no enforced ownership):**

.. code-block:: cpp

    std::string* ptr;
    {
        std::string s = "hello";
        ptr = &s;
    }  // s destroyed
    std::cout << *ptr;  // Undefined behavior: dangling pointer

Rust's ownership model makes the single-owner rule explicit. When you assign a
heap-allocated value to another variable, ownership transfers (moves) to the new
variable. The original variable becomes invalid, and the compiler will reject any
attempt to use it. This eliminates use-after-move bugs at compile time rather than
discovering them through crashes or sanitizers at runtime.

**Rust (ownership enforced):**

.. code-block:: rust

    let s1 = String::from("hello");  // s1 owns the String
    let s2 = s1;                      // ownership moved to s2
    // println!("{}", s1);            // Error: s1 no longer valid
    println!("{}", s2);               // OK: s2 is the owner

Move vs Copy
------------

:Source: `src/rust/ownership <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/ownership>`_

C++ defaults to copying on assignment, which can be expensive for large objects.
To move, you must explicitly use ``std::move()``. This opt-in approach means
programmers often forget to move when they should, leading to unnecessary copies.
After a move, the source object is in a "valid but unspecified state" - you can
still accidentally use it, leading to subtle bugs.

**C++:**

.. code-block:: cpp

    std::string s1 = "hello";
    std::string s2 = s1;           // Copy (expensive)
    std::string s3 = std::move(s1); // Move (s1 now empty)

Rust inverts this default: assignment moves by default for types that manage heap
memory (like ``String``, ``Vec``, ``Box``). If you want a copy, you must explicitly
call ``.clone()``. This makes the cost of copying visible in the code. For simple
stack-only types like integers and floats, Rust implements the ``Copy`` trait,
allowing implicit copying since it's cheap. After a move in Rust, the source
variable is completely invalid - not just empty - so the compiler catches any
accidental use.

**Rust:**

.. code-block:: rust

    let s1 = String::from("hello");
    let s2 = s1;              // Move (s1 invalidated)
    let s3 = s2.clone();      // Explicit copy

    // Copy types (primitives, tuples of Copy types)
    let x = 5;
    let y = x;                // Copy (both valid)
    println!("{} {}", x, y);  // OK

Borrowing vs C++ References
---------------------------

:Source: `src/rust/borrowing <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/borrowing>`_

Rust borrowing (``&T``, ``&mut T``) looks like C++ references but with crucial
differences enforced at compile time:

+---------------------------+----------------------------------+----------------------------------+
| Aspect                    | C++ References                   | Rust Borrowing                   |
+===========================+==================================+==================================+
| Validity                  | Not checked (can dangle)         | Compiler-enforced (never dangle) |
+---------------------------+----------------------------------+----------------------------------+
| Multiple readers          | Allowed (no enforcement)         | Allowed (``&T``)                 |
+---------------------------+----------------------------------+----------------------------------+
| Single writer             | Not enforced                     | Enforced (``&mut T`` exclusive)  |
+---------------------------+----------------------------------+----------------------------------+
| Aliasing + mutation       | Allowed (causes bugs)            | Forbidden at compile time        |
+---------------------------+----------------------------------+----------------------------------+
| Null                      | Possible (undefined behavior)    | Not possible                     |
+---------------------------+----------------------------------+----------------------------------+

One of the most insidious bugs in C++ occurs when you hold a reference to an element
inside a container, then modify the container in a way that invalidates that reference.
The classic example is holding a reference to a vector element, then pushing to the
vector. If the vector reallocates, your reference now points to freed memory. The
C++ compiler cannot detect this because it doesn't track the relationship between
the reference and the container.

**C++ (compiles but has data race potential):**

.. code-block:: cpp

    void process(std::vector<int>& vec, int& elem) {
        vec.push_back(42);  // May invalidate elem!
        std::cout << elem;  // Undefined behavior if reallocated
    }

    int main() {
        std::vector<int> v = {1, 2, 3};
        process(v, v[0]);  // Dangerous: elem may dangle after push_back
    }

Rust's borrow checker understands that when you borrow an element from a vector,
you're borrowing from the vector itself. It won't let you mutate the vector while
that borrow exists. This prevents iterator invalidation, a entire class of bugs
that plagues C++ codebases. The error message clearly explains why the code is
rejected: you cannot have a mutable borrow of the vector while an immutable borrow
of its contents exists.

**Rust (compile error prevents the bug):**

.. code-block:: rust

    fn process(vec: &mut Vec<i32>, elem: &i32) {
        vec.push(42);       // Error: cannot borrow `vec` as mutable
        println!("{}", elem); // because `elem` is borrowed from `vec`
    }

    fn main() {
        let mut v = vec![1, 2, 3];
        let elem = &v[0];
        // process(&mut v, elem);  // Won't compile
    }

Borrowing Rules
---------------

:Source: `src/rust/borrowing <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/borrowing>`_

The borrow checker enforces two fundamental rules at compile time that prevent
data races and aliasing bugs:

1. You can have either one mutable reference OR any number of immutable references
2. References must always be valid (no dangling)

These rules implement a reader-writer lock pattern at compile time. Multiple readers
can access data simultaneously (shared/immutable borrows), but a writer needs
exclusive access (mutable borrow). Unlike runtime locks, violations are caught
during compilation with zero runtime overhead. The borrow checker also tracks when
borrows end, so you can have a mutable borrow after immutable borrows go out of scope.

.. code-block:: rust

    let mut s = String::from("hello");

    // Multiple immutable borrows: OK
    let r1 = &s;
    let r2 = &s;
    println!("{} {}", r1, r2);

    // Mutable borrow after immutable borrows end: OK
    let r3 = &mut s;
    r3.push_str(" world");

    // Simultaneous mutable and immutable: ERROR
    // let r4 = &s;
    // let r5 = &mut s;  // Error: cannot borrow as mutable

Dangling References
-------------------

:Source: `src/rust/lifetimes <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/lifetimes>`_

Returning a reference to a local variable is a classic C++ bug. The function creates
a local string on the stack, returns a reference to it, and then the stack frame
is destroyed. The caller receives a reference to garbage memory. Most compilers
will warn about this, but it's not a hard error, and more complex cases (like
returning a reference through multiple function calls) often escape detection.

**C++ (compiles but undefined behavior):**

.. code-block:: cpp

    std::string& get_string() {
        std::string s = "hello";
        return s;  // Dangling reference!
    }

Rust's borrow checker performs lifetime analysis to prove that all references are
valid. It understands that a reference to a local variable cannot outlive the
function call. The error message is clear: the local variable ``s`` does not live
long enough to be returned as a reference. The solution is to return owned data,
transferring ownership to the caller, or to borrow from data the caller provides.

**Rust (compile error):**

.. code-block:: rust

    // Won't compile: cannot return reference to local variable
    fn get_string() -> &String {
        let s = String::from("hello");
        &s  // Error: `s` does not live long enough
    }

    // Solution: return owned data
    fn get_string() -> String {
        String::from("hello")  // Ownership transferred to caller
    }

Lifetime Annotations
--------------------

:Source: `src/rust/lifetimes <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/lifetimes>`_

When a function takes multiple references and returns a reference, the compiler
needs to know which input the output is derived from. This determines how long
the returned reference is valid. C++ has no way to express this relationship,
so the compiler cannot verify that the returned reference won't dangle.

**C++ (no lifetime tracking):**

.. code-block:: cpp

    // C++ doesn't track which input the return value depends on
    const std::string& longer(const std::string& a, const std::string& b) {
        return a.size() > b.size() ? a : b;
    }

    int main() {
        const std::string& result = longer("short", "longer string");
        // Both temporaries destroyed, result is dangling!
    }

Rust's lifetime annotations (``'a``) are a way to tell the compiler about the
relationship between input and output references. The syntax ``fn longer<'a>(a: &'a str, b: &'a str) -> &'a str``
means "the returned reference will be valid as long as both input references are
valid." The compiler uses this information to ensure callers don't use the result
after either input has been dropped. Lifetimes don't change how long values live;
they're annotations that help the compiler verify your code is safe.

**Rust (explicit lifetime annotation):**

.. code-block:: rust

    // 'a means: returned reference lives as long as both inputs
    fn longer<'a>(a: &'a str, b: &'a str) -> &'a str {
        if a.len() > b.len() { a } else { b }
    }

    fn main() {
        let s1 = String::from("short");
        let s2 = String::from("longer string");
        let result = longer(&s1, &s2);  // OK: s1 and s2 outlive result
        println!("{}", result);
    }

Lifetime Elision
----------------

:Source: `src/rust/lifetimes <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/lifetimes>`_

Writing lifetime annotations everywhere would be tedious, so Rust has elision rules
that let you omit them in common patterns. The compiler applies these rules
automatically, inserting the lifetimes for you. If the rules don't apply (like
when you have multiple input references and return a reference), you must write
explicit annotations. Understanding elision helps you know when annotations are
needed and makes reading Rust code easier.

1. Each input reference gets its own lifetime
2. If there's exactly one input lifetime, it's assigned to all outputs
3. If there's ``&self`` or ``&mut self``, its lifetime is assigned to outputs

.. code-block:: rust

    // These are equivalent:
    fn first_word(s: &str) -> &str { ... }
    fn first_word<'a>(s: &'a str) -> &'a str { ... }

    // These are equivalent:
    fn get_name(&self) -> &str { ... }
    fn get_name<'a>(&'a self) -> &'a str { ... }

Structs with References
-----------------------

:Source: `src/rust/lifetimes <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/lifetimes>`_

Storing references in structs is dangerous in C++ because there's no guarantee the
referenced data outlives the struct. A common bug is constructing a struct with a
reference to a temporary, which is destroyed immediately, leaving the struct with
a dangling reference. The C++ compiler may not warn about this, especially when
the temporary's lifetime is extended in some cases but not others.

**C++ (no compile-time guarantee):**

.. code-block:: cpp

    struct Excerpt {
        const std::string& text;  // Reference member
        Excerpt(const std::string& t) : text(t) {}
    };

    int main() {
        Excerpt e("temp");  // Dangling: temporary destroyed
        std::cout << e.text;  // Undefined behavior
    }

In Rust, a struct containing a reference must declare a lifetime parameter. This
lifetime represents "the struct cannot outlive the data it references." The compiler
enforces this constraint at every use site. You cannot create an ``Excerpt`` that
outlives the string it borrows from. This makes reference-holding structs safe to
use, enabling patterns like zero-copy parsing where you return structs that borrow
from the input data.

**Rust (lifetime enforced):**

.. code-block:: rust

    struct Excerpt<'a> {
        text: &'a str,  // Reference must live at least as long as struct
    }

    fn main() {
        let novel = String::from("Call me Ishmael...");
        let excerpt = Excerpt { text: &novel };
        println!("{}", excerpt.text);
    }  // novel outlives excerpt, so this is safe

Static Lifetime
---------------

:Source: `src/rust/lifetimes <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/lifetimes>`_

The ``'static`` lifetime is a special lifetime that means "valid for the entire
program duration." String literals in both C++ and Rust have static storage duration
- they're embedded in the binary and exist from program start to end. In Rust,
string literals have type ``&'static str``. The ``'static`` bound is also used
in trait bounds to indicate a type contains no non-static references, which is
required for spawning threads (since the thread might outlive the current scope).

.. code-block:: rust

    // String literals are &'static str
    fn get_greeting() -> &'static str {
        "Hello, world!"  // OK: string literal has 'static lifetime
    }

    // Static variables also have 'static lifetime
    static GREETING: &str = "Hello";

Interior Mutability
-------------------

:Source: `src/rust/smart_pointers <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/smart_pointers>`_

Sometimes you need to mutate data through a shared (immutable) reference. In C++,
the ``mutable`` keyword allows modifying members even in const methods - commonly
used for caches, lazy initialization, or reference counting. This bypasses const
correctness and can lead to data races if not carefully managed.

**C++ (mutable keyword):**

.. code-block:: cpp

    class Counter {
        mutable int count = 0;  // Can modify even in const methods
    public:
        void increment() const { ++count; }
        int get() const { return count; }
    };

Rust provides interior mutability through types like ``RefCell<T>`` and ``Cell<T>``.
These types move borrow checking from compile time to runtime. ``RefCell`` tracks
borrows dynamically and will panic if you violate the borrowing rules (e.g., two
mutable borrows at once). This is useful for patterns like the observer pattern,
graph structures with cycles, or mock objects in tests. Use interior mutability
sparingly - compile-time checking is preferable when possible.

**Rust (RefCell for interior mutability):**

.. code-block:: rust

    use std::cell::RefCell;

    struct Counter {
        count: RefCell<i32>,
    }

    impl Counter {
        fn new() -> Self {
            Counter { count: RefCell::new(0) }
        }

        fn increment(&self) {  // &self, not &mut self
            *self.count.borrow_mut() += 1;
        }

        fn get(&self) -> i32 {
            *self.count.borrow()
        }
    }

Common Ownership Patterns
-------------------------

:Source: `src/rust/ownership <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/ownership>`_

Understanding when to take ownership versus borrow is key to writing idiomatic Rust.
Take ownership when the function needs to store the value, transfer it elsewhere,
or the caller won't need it anymore. Borrow when you only need to read or temporarily
modify the value. These patterns become second nature with practice.

**Taking ownership (consuming):**

When a function takes ownership, the caller gives up the value permanently. This is
appropriate when the function will store the value in a data structure, return it
as part of a larger value, or when the caller is done with it. The value is dropped
when the function ends (unless returned or stored).

.. code-block:: rust

    fn consume(s: String) {
        println!("{}", s);
    }  // s dropped here

    let s = String::from("hello");
    consume(s);
    // println!("{}", s);  // Error: s was moved

**Borrowing (non-consuming):**

Borrowing with ``&T`` lets a function read data without taking ownership. The caller
retains ownership and can continue using the value after the function returns. This
is the most common pattern for functions that only need to inspect data.

.. code-block:: rust

    fn borrow(s: &String) {
        println!("{}", s);
    }

    let s = String::from("hello");
    borrow(&s);
    println!("{}", s);  // OK: s still valid

**Mutable borrowing:**

Mutable borrowing with ``&mut T`` allows temporary exclusive access to modify data.
The caller retains ownership but cannot access the value while it's mutably borrowed.
This is used when a function needs to modify data in place without taking ownership.

.. code-block:: rust

    fn modify(s: &mut String) {
        s.push_str(" world");
    }

    let mut s = String::from("hello");
    modify(&mut s);
    println!("{}", s);  // "hello world"

**Returning ownership:**

Functions can create values and transfer ownership to the caller. This is how
constructors work in Rust (conventionally named ``new``). The caller becomes the
owner and is responsible for the value's lifetime.

.. code-block:: rust

    fn create() -> String {
        String::from("created")  // Ownership transferred to caller
    }

    let s = create();  // s owns the String
