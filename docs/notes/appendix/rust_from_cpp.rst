=======================
Rust for C++ Developers
=======================

.. meta::
   :description: Learn Rust from a C++ background with side-by-side code comparisons. Covers ownership, borrowing, lifetimes, traits, generics, error handling, smart pointers, threads, and modules with practical examples.
   :keywords: Rust, C++, learn Rust, C++ to Rust, Rust tutorial, ownership, borrowing, lifetimes, traits, generics, Result, Option, smart pointers, Arc, Mutex, Rc, RefCell, modules, cargo, rustc

.. contents:: Table of Contents
    :backlinks: none

This appendix provides a mapping between C++ and Rust concepts for developers
transitioning from C++ to Rust. Each section shows the C++ pattern alongside
its Rust equivalent.

Hello World
-----------

:Source: `src/rust/hello <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/hello>`_

The classic first program. In C++, you need to include ``<iostream>`` and use
``std::cout`` for output. Rust provides the ``println!`` macro which handles
formatting and automatically appends a newline. Note that Rust's ``main``
function doesn't return a value by default (it implicitly returns ``()``).

**C++:**

.. code-block:: cpp

    #include <iostream>

    int main() {
        std::cout << "Hello, World!" << std::endl;
        return 0;
    }

**Rust:**

.. code-block:: rust

    fn main() {
        println!("Hello, World!");
    }

Variables and Mutability
------------------------

:Source: `src/rust/variables <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/variables>`_

One of the most fundamental differences between C++ and Rust is the default
mutability of variables. In C++, variables are mutable by default and you use
``const`` to make them immutable. Rust takes the opposite approach: variables
are immutable by default, and you must explicitly use the ``mut`` keyword to
allow mutation. This design choice helps prevent accidental modifications and
makes code intent clearer.

**C++:**

.. code-block:: cpp

    int x = 5;           // mutable by default
    x = 10;              // OK
    const int y = 5;     // immutable
    // y = 10;           // error

**Rust:**

.. code-block:: rust

    let x = 5;           // immutable by default
    // x = 10;           // error
    let mut y = 5;       // mutable
    y = 10;              // OK

References and Borrowing
------------------------

:Source: `src/rust/borrowing <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/borrowing>`_

C++ references are simple aliases to existing variables, and the programmer is
responsible for ensuring they don't outlive the data they reference. Rust's
borrowing system is more sophisticated: the compiler enforces that you can have
either one mutable reference OR any number of immutable references to a value,
but never both at the same time. This rule, checked at compile time, prevents
data races and iterator invalidation bugs that are common in C++.

**C++:**

.. code-block:: cpp

    void modify(int& x) { x += 1; }
    void read(const int& x) { std::cout << x; }

    int main() {
        int val = 5;
        modify(val);     // mutable reference
        read(val);       // const reference
    }

**Rust:**

.. code-block:: rust

    fn modify(x: &mut i32) { *x += 1; }
    fn read(x: &i32) { println!("{}", x); }

    fn main() {
        let mut val = 5;
        modify(&mut val);  // mutable borrow
        read(&val);        // immutable borrow
    }

Ownership and Move Semantics
----------------------------

:Source: `src/rust/ownership <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/ownership>`_

C++ has copy semantics by default: assigning one variable to another creates a
copy of the data. Move semantics were added in C++11 via ``std::move``, but
using a moved-from object is undefined behavior that the compiler won't catch.
Rust flips this: move semantics are the default for types that manage resources.
When you assign a ``String`` to another variable, ownership transfers and the
original variable becomes invalid. The compiler enforces this, preventing
use-after-move bugs entirely. For types that are cheap to copy (like integers),
Rust uses the ``Copy`` trait to enable implicit copying.

**C++:**

.. code-block:: cpp

    #include <string>
    #include <utility>

    int main() {
        std::string s1 = "hello";
        std::string s2 = s1;              // copy
        std::string s3 = std::move(s1);   // move, s1 is now empty
    }

**Rust:**

.. code-block:: rust

    fn main() {
        let s1 = String::from("hello");
        let s2 = s1.clone();  // explicit clone
        let s3 = s1;          // move, s1 is no longer valid
        // println!("{}", s1); // error: value borrowed after move
    }

Structs and Methods
-------------------

:Source: `src/rust/structs <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/structs>`_

Both languages support defining custom types with associated methods. In C++,
methods are defined inside the struct/class body. Rust separates data definition
(``struct``) from behavior (``impl`` blocks), which allows you to add methods
to types in different modules or even extend types from external crates. Rust
doesn't have constructors; instead, you define associated functions like ``new``
by convention. The ``&self`` parameter is similar to C++'s implicit ``this``
pointer but is explicit in Rust.

**C++:**

.. code-block:: cpp

    struct Point {
        double x, y;
        Point(double x, double y) : x(x), y(y) {}
        double distance() const {
            return std::sqrt(x*x + y*y);
        }
    };

**Rust:**

.. code-block:: rust

    struct Point {
        x: f64,
        y: f64,
    }

    impl Point {
        fn new(x: f64, y: f64) -> Self {
            Point { x, y }
        }
        fn distance(&self) -> f64 {
            (self.x * self.x + self.y * self.y).sqrt()
        }
    }

this vs self
------------

:Source: `src/rust/self_this <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/self_this>`_

In C++, ``this`` is an implicit pointer to the current object, available in all
non-static member functions. In Rust, ``self`` must be explicitly declared as
the first parameter. Rust provides three forms: ``self`` (takes ownership),
``&self`` (immutable borrow), and ``&mut self`` (mutable borrow). This explicit
declaration makes the method's relationship with the object clear and integrates
with the ownership system. ``Self`` (capital S) is a type alias for the
implementing type.

**C++:**

.. code-block:: cpp

    class Counter {
        int value = 0;
    public:
        void increment() {           // 'this' is implicit
            this->value++;           // explicit 'this->' optional
            value++;                 // same as above
        }
        int get() const {            // const method: 'this' is const
            return this->value;
        }
        Counter* get_this() {
            return this;             // return pointer to self
        }
    };

**Rust:**

.. code-block:: rust

    struct Counter {
        value: i32,
    }

    impl Counter {
        fn new() -> Self {                    // Self = Counter
            Counter { value: 0 }
        }
        fn increment(&mut self) {             // mutable borrow of self
            self.value += 1;
        }
        fn get(&self) -> i32 {                // immutable borrow of self
            self.value
        }
        fn into_value(self) -> i32 {          // takes ownership of self
            self.value                        // self is consumed
        }
    }

    fn main() {
        let mut c = Counter::new();
        c.increment();
        println!("{}", c.get());
        let v = c.into_value();  // c is moved, no longer valid
    }

**Rust method receiver types:**

.. code-block:: rust

    impl MyStruct {
        fn by_ref(&self) {}           // borrows: &MyStruct
        fn by_mut(&mut self) {}       // mutably borrows: &mut MyStruct
        fn by_value(self) {}          // takes ownership: MyStruct
        fn by_box(self: Box<Self>) {} // takes boxed self
        fn by_rc(self: Rc<Self>) {}   // takes Rc self
    }

RAII and Destructors
--------------------

:Source: `src/rust/raii <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/raii>`_

RAII (Resource Acquisition Is Initialization) is fundamental to both languages.
In C++, you implement destructors to release resources when objects go out of
scope. Rust uses the ``Drop`` trait for the same purpose. The key difference is
that Rust's ownership system guarantees exactly when destructors run, and the
compiler prevents use-after-free bugs. Rust also provides ``std::mem::drop()``
to explicitly drop a value before its scope ends, and ``std::mem::forget()`` to
prevent the destructor from running (useful for FFI).

**C++:**

.. code-block:: cpp

    #include <iostream>

    class FileHandle {
        FILE* file;
    public:
        FileHandle(const char* path) : file(fopen(path, "r")) {
            std::cout << "File opened\n";
        }
        ~FileHandle() {
            if (file) {
                fclose(file);
                std::cout << "File closed\n";
            }
        }
        // Rule of five: also need copy/move constructors and assignments
    };

    int main() {
        FileHandle fh("test.txt");
        // destructor called automatically at end of scope
    }

**Rust:**

.. code-block:: rust

    use std::fs::File;

    struct FileHandle {
        file: Option<File>,
        name: String,
    }

    impl FileHandle {
        fn new(path: &str) -> std::io::Result<Self> {
            println!("File opened");
            Ok(FileHandle {
                file: Some(File::open(path)?),
                name: path.to_string(),
            })
        }
    }

    impl Drop for FileHandle {
        fn drop(&mut self) {
            println!("File closed: {}", self.name);
            // File is automatically closed when dropped
        }
    }

    fn main() {
        let _fh = FileHandle::new("test.txt");
        // Drop::drop called automatically at end of scope
    }

**Rust explicit drop:**

.. code-block:: rust

    fn main() {
        let v = vec![1, 2, 3];
        drop(v);  // explicitly drop before scope ends
        // v is no longer valid here
    }

Strings
-------

:Source: `src/rust/strings <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/strings>`_

C++ has ``std::string`` for owned strings and ``std::string_view`` (C++17) for
borrowed string slices. Rust distinguishes between ``String`` (owned, heap-
allocated, growable) and ``&str`` (borrowed string slice). This distinction is
enforced by the type system: functions that don't need ownership should take
``&str``, allowing them to accept both ``String`` references and string literals.
Rust strings are always valid UTF-8, unlike C++ strings which are just byte
sequences.

**C++:**

.. code-block:: cpp

    #include <string>
    #include <string_view>

    void print(std::string_view s) {  // borrowed, doesn't copy
        std::cout << s << "\n";
    }

    int main() {
        std::string owned = "hello";       // owned string
        owned += " world";                 // mutable
        print(owned);                      // implicit conversion to string_view
        print("literal");                  // works with literals too
    }

**Rust:**

.. code-block:: rust

    fn print(s: &str) {  // borrowed string slice
        println!("{}", s);
    }

    fn main() {
        let owned = String::from("hello");  // owned String
        let owned = owned + " world";       // concatenation (consumes owned)
        print(&owned);                      // borrow as &str
        print("literal");                   // &str literal

        // String methods
        let s = String::from("  hello  ");
        let trimmed = s.trim();             // returns &str
        let upper = s.to_uppercase();       // returns new String
    }

Type Casting
------------

:Source: `src/rust/casting <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/casting>`_

C++ provides multiple cast operators with different safety levels:
``static_cast``, ``dynamic_cast``, ``const_cast``, and ``reinterpret_cast``.
Rust uses the ``as`` keyword for primitive type conversions and the ``From``/
``Into`` traits for more complex conversions. Rust's approach is more explicit
and the compiler prevents many unsafe conversions. For fallible conversions,
Rust uses ``TryFrom``/``TryInto`` which return ``Result``.

**C++:**

.. code-block:: cpp

    int main() {
        // Numeric conversions
        double d = 3.14;
        int i = static_cast<int>(d);       // truncates to 3

        // Pointer casts
        void* ptr = &i;
        int* ip = static_cast<int*>(ptr);

        // Polymorphic downcast (requires RTTI)
        Base* base = new Derived();
        Derived* derived = dynamic_cast<Derived*>(base);
    }

**Rust:**

.. code-block:: rust

    fn main() {
        // Numeric conversions with 'as'
        let d: f64 = 3.14;
        let i: i32 = d as i32;             // truncates to 3

        // Safe conversions with From/Into
        let s: String = String::from("hello");
        let i: i64 = i32::MAX.into();      // widening is safe

        // Fallible conversions with TryFrom/TryInto
        let big: i64 = 1000;
        let small: Result<i8, _> = big.try_into();  // Err: overflow

        // Pointer casts (unsafe)
        let ptr: *const i32 = &i;
        let addr: usize = ptr as usize;
    }

Compile-Time Computation (constexpr)
------------------------------------

:Source: `src/rust/const_fn <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/const_fn>`_

C++ ``constexpr`` allows functions and variables to be evaluated at compile
time. Rust has ``const fn`` for compile-time function evaluation and ``const``
for compile-time constants. Rust's const evaluation is more restricted than
C++'s constexpr but is being expanded with each edition. Both languages use
these features to move computation from runtime to compile time, enabling
optimizations and compile-time validation.

**C++:**

.. code-block:: cpp

    constexpr int factorial(int n) {
        return n <= 1 ? 1 : n * factorial(n - 1);
    }

    constexpr int fact_5 = factorial(5);  // computed at compile time

    // constexpr if (C++17)
    template<typename T>
    auto process(T value) {
        if constexpr (std::is_integral_v<T>) {
            return value * 2;
        } else {
            return value;
        }
    }

**Rust:**

.. code-block:: rust

    const fn factorial(n: u64) -> u64 {
        match n {
            0 | 1 => 1,
            _ => n * factorial(n - 1),
        }
    }

    const FACT_5: u64 = factorial(5);  // computed at compile time

    // const generics
    fn create_array<const N: usize>() -> [i32; N] {
        [0; N]
    }

    fn main() {
        let arr = create_array::<5>();  // [0, 0, 0, 0, 0]
    }

Enums and Pattern Matching
--------------------------

:Source: `src/rust/enums <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/enums>`_

C++ enums are essentially named integers. To create a sum type (a type that can
be one of several variants, each potentially holding different data), you need
``std::variant``. Rust enums are true algebraic data types: each variant can
hold different types and amounts of data. Combined with ``match`` expressions,
which must handle all variants exhaustively, Rust enums provide a powerful and
type-safe way to model state machines, error handling, and optional values.

**C++:**

.. code-block:: cpp

    #include <variant>
    #include <string>

    enum class MessageType { Quit, Move, Write };

    struct Move { int x, y; };
    struct Write { std::string msg; };

    using Message = std::variant<std::monostate, Move, Write>;

**Rust:**

.. code-block:: rust

    enum Message {
        Quit,
        Move { x: i32, y: i32 },
        Write(String),
    }

    fn process(msg: Message) {
        match msg {
            Message::Quit => println!("Quit"),
            Message::Move { x, y } => println!("Move to ({}, {})", x, y),
            Message::Write(s) => println!("Write: {}", s),
        }
    }

Option and Result (Error Handling)
----------------------------------

:Source: `src/rust/error_handling <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/error_handling>`_

C++ traditionally uses exceptions for error handling, though ``std::optional``
(C++17) and ``std::expected`` (C++23) provide alternatives. Rust doesn't have
exceptions. Instead, it uses ``Option<T>`` for values that might be absent and
``Result<T, E>`` for operations that can fail. These are regular enum types,
and the compiler ensures you handle both cases. The ``?`` operator provides
ergonomic error propagation, automatically returning early if an error occurs.

**C++:**

.. code-block:: cpp

    #include <optional>
    #include <expected>  // C++23

    std::optional<int> find_value(int key) {
        if (key > 0) return key * 2;
        return std::nullopt;
    }

    std::expected<int, std::string> divide(int a, int b) {
        if (b == 0) return std::unexpected("division by zero");
        return a / b;
    }

**Rust:**

.. code-block:: rust

    fn find_value(key: i32) -> Option<i32> {
        if key > 0 { Some(key * 2) } else { None }
    }

    fn divide(a: i32, b: i32) -> Result<i32, String> {
        if b == 0 {
            Err("division by zero".to_string())
        } else {
            Ok(a / b)
        }
    }

    fn main() {
        // Using ? operator for error propagation
        if let Some(v) = find_value(5) {
            println!("Found: {}", v);
        }
    }

Generics and Traits
-------------------

:Source: `src/rust/generics <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/generics>`_

C++ templates use duck typing: any type that supports the required operations
will work, but errors only appear at instantiation time and can be cryptic.
C++20 concepts improve this by allowing explicit constraints. Rust generics
require explicit trait bounds from the start. The compiler checks that the
generic code only uses operations provided by the specified traits, giving
clear error messages at the definition site rather than the call site.

**C++:**

.. code-block:: cpp

    template<typename T>
    T max(T a, T b) {
        return (a > b) ? a : b;
    }

    // With concepts (C++20)
    template<typename T>
    concept Comparable = requires(T a, T b) { a > b; };

    template<Comparable T>
    T max_concept(T a, T b) {
        return (a > b) ? a : b;
    }

**Rust:**

.. code-block:: rust

    fn max<T: PartialOrd>(a: T, b: T) -> T {
        if a > b { a } else { b }
    }

    // Or with where clause
    fn max_where<T>(a: T, b: T) -> T
    where
        T: PartialOrd,
    {
        if a > b { a } else { b }
    }

Iterators
---------

:Source: `src/rust/iterators <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/iterators>`_

Both languages support iterator-based programming, but with different
philosophies. C++ iterators are pointer-like objects that you manipulate
directly with algorithms from ``<algorithm>``. Rust iterators are lazy and
chainable: operations like ``filter`` and ``map`` don't execute until you
consume the iterator (e.g., with ``collect``). This allows the compiler to
optimize the entire chain as a single loop, achieving zero-cost abstraction.

**C++:**

.. code-block:: cpp

    #include <vector>
    #include <algorithm>
    #include <numeric>

    int main() {
        std::vector<int> v = {1, 2, 3, 4, 5};

        // Transform and filter
        std::vector<int> result;
        for (int x : v) {
            if (x % 2 == 0) {
                result.push_back(x * 2);
            }
        }

        // Sum
        int sum = std::accumulate(v.begin(), v.end(), 0);
    }

**Rust:**

.. code-block:: rust

    fn main() {
        let v = vec![1, 2, 3, 4, 5];

        // Transform and filter (lazy, chained)
        let result: Vec<i32> = v.iter()
            .filter(|&&x| x % 2 == 0)
            .map(|&x| x * 2)
            .collect();

        // Sum
        let sum: i32 = v.iter().sum();
    }

Smart Pointers
--------------

:Source: `src/rust/smart_pointers <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/smart_pointers>`_

Both languages provide smart pointers for automatic memory management. C++ has
``unique_ptr`` (single ownership), ``shared_ptr`` (reference counted), and
``weak_ptr`` (non-owning reference). Rust's equivalents are ``Box<T>`` (heap
allocation with single ownership), ``Rc<T>`` (reference counted for single-
threaded use), and ``Arc<T>`` (atomic reference counted for multi-threaded use).
Rust also provides ``RefCell<T>`` for interior mutability, allowing mutation
through shared references with runtime borrow checking.

**C++:**

.. code-block:: cpp

    #include <memory>

    int main() {
        auto unique = std::make_unique<int>(42);
        auto shared = std::make_shared<int>(42);
        std::weak_ptr<int> weak = shared;
    }

**Rust:**

.. code-block:: rust

    use std::rc::Rc;
    use std::cell::RefCell;

    fn main() {
        let boxed = Box::new(42);           // unique ownership (heap)
        let shared = Rc::new(42);           // reference counted
        let shared_mut = Rc::new(RefCell::new(42));  // interior mutability
    }

Threads and Concurrency
-----------------------

:Source: `src/rust/threads <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/threads>`_

C++ provides ``std::thread`` and synchronization primitives like ``std::mutex``,
but the compiler cannot prevent data races. Rust's ownership system extends to
concurrency: the ``Send`` and ``Sync`` traits mark types that can be safely
transferred between or shared across threads. The compiler enforces these
constraints, making data races impossible in safe Rust. Shared mutable state
requires explicit synchronization with ``Arc<Mutex<T>>``, making the locking
visible in the type system.

**C++:**

.. code-block:: cpp

    #include <thread>
    #include <mutex>

    int main() {
        int counter = 0;
        std::mutex mtx;

        auto worker = [&]() {
            std::lock_guard<std::mutex> lock(mtx);
            counter++;
        };

        std::thread t1(worker);
        std::thread t2(worker);
        t1.join();
        t2.join();
    }

**Rust:**

.. code-block:: rust

    use std::sync::{Arc, Mutex};
    use std::thread;

    fn main() {
        let counter = Arc::new(Mutex::new(0));

        let handles: Vec<_> = (0..2).map(|_| {
            let counter = Arc::clone(&counter);
            thread::spawn(move || {
                let mut num = counter.lock().unwrap();
                *num += 1;
            })
        }).collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

Closures
--------

:Source: `src/rust/closures <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/closures>`_

Both languages support closures (anonymous functions that capture variables from
their environment). C++ uses explicit capture lists: ``[&]`` for by-reference,
``[=]`` for by-value, or individual variables. Rust closures automatically infer
how to capture each variable (by reference, mutable reference, or by value)
based on usage. The ``move`` keyword forces all captures to be by value,
transferring ownership into the closure. This is essential for closures that
outlive their creation scope, such as those passed to ``thread::spawn``.

**C++:**

.. code-block:: cpp

    #include <functional>

    int main() {
        int x = 10;
        auto by_ref = [&x]() { return x + 1; };      // capture by reference
        auto by_val = [x]() { return x + 1; };       // capture by value
        auto by_move = [x = std::move(x)]() { return x; };  // capture by move
    }

**Rust:**

.. code-block:: rust

    fn main() {
        let x = 10;
        let by_ref = || x + 1;                    // borrows x
        let by_move = move || x + 1;              // takes ownership of x

        let mut y = 10;
        let mut by_mut = || { y += 1; y };        // mutably borrows y
    }

Vectors and Collections
-----------------------

:Source: `src/rust/collections <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/collections>`_

Both languages provide similar collection types. C++'s ``std::vector`` maps to
Rust's ``Vec<T>``, ``std::map`` (ordered, tree-based) maps to ``BTreeMap``, and
``std::unordered_map`` (hash-based) maps to ``HashMap``. The main difference is
that Rust collections integrate with the ownership system: you can't have
multiple mutable references to elements, and iterators are invalidated if you
modify the collection. This prevents iterator invalidation bugs at compile time.

**C++:**

.. code-block:: cpp

    #include <vector>
    #include <map>
    #include <unordered_map>

    int main() {
        std::vector<int> vec = {1, 2, 3};
        vec.push_back(4);

        std::map<std::string, int> ordered;
        ordered["one"] = 1;

        std::unordered_map<std::string, int> hash;
        hash["one"] = 1;
    }

**Rust:**

.. code-block:: rust

    use std::collections::{BTreeMap, HashMap};

    fn main() {
        let mut vec = vec![1, 2, 3];
        vec.push(4);

        let mut ordered = BTreeMap::new();
        ordered.insert("one", 1);

        let mut hash = HashMap::new();
        hash.insert("one", 1);
    }

Modules
-------

:Source: `src/rust/modules <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/modules>`_

C++20 introduced modules to replace the textual inclusion model of header files,
improving compilation times and encapsulation. Rust has had a module system from
the beginning. Modules are declared with ``mod`` and items are made public with
``pub``. The ``use`` keyword brings items into scope. Rust's module system is
hierarchical: a file ``src/foo.rs`` or directory ``src/foo/mod.rs`` defines a
module named ``foo``. This structure makes the relationship between files and
modules explicit and predictable.

**C++ (C++20 modules):**

.. code-block:: text

    project/
    ├── CMakeLists.txt
    ├── math.cppm        # module interface unit
    └── main.cpp

.. code-block:: cpp

    // math.cppm (module interface)
    export module math;

    export int add(int a, int b) { return a + b; }
    export int multiply(int a, int b) { return a * b; }

    // main.cpp
    import math;

    int main() {
        int sum = add(2, 3);
        int product = multiply(2, 3);
    }

**Rust:**

.. code-block:: text

    project/
    ├── Cargo.toml
    └── src/
        ├── main.rs
        └── math.rs      # module file

.. code-block:: rust

    // src/math.rs
    pub fn add(a: i32, b: i32) -> i32 { a + b }
    pub fn multiply(a: i32, b: i32) -> i32 { a * b }

    // src/main.rs
    mod math;

    fn main() {
        let sum = math::add(2, 3);
        let product = math::multiply(2, 3);
    }

**Rust nested modules (directory style):**

.. code-block:: text

    project/
    └── src/
        ├── main.rs
        └── outer/
            ├── mod.rs       # declares submodules
            └── inner.rs

.. code-block:: rust

    // src/outer/mod.rs
    pub mod inner;

    // src/outer/inner.rs
    pub fn greet() { println!("Hello!"); }

    // src/main.rs
    mod outer;

    fn main() {
        outer::inner::greet();
    }

**Rust library with re-exports:**

.. code-block:: text

    my_lib/
    ├── Cargo.toml
    └── src/
        ├── lib.rs           # library root
        ├── internal.rs      # private module
        └── public_api.rs

.. code-block:: rust

    // src/lib.rs - re-export for cleaner API
    mod internal;
    pub mod public_api;
    pub use internal::public_function;  // re-export at crate root
