=======
Strings
=======

.. meta::
   :description: Rust String and &str compared to C++ std::string and string_view. Covers owned strings, string slices, UTF-8, and common operations.
   :keywords: Rust, String, str, string slice, UTF-8, std::string, string_view, C++ comparison

.. contents:: Table of Contents
    :backlinks: none

:Source: `src/rust/strings <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/strings>`_

Rust has two main string types: ``String`` (owned, heap-allocated, growable) and
``&str`` (borrowed string slice). This is similar to C++'s ``std::string`` and
``std::string_view``, but Rust strings are always valid UTF-8.

String vs &str
--------------

**C++:**

.. code-block:: cpp

    #include <string>
    #include <string_view>

    void print(std::string_view s) {  // borrowed, no copy
      std::cout << s << "\n";
    }

    int main() {
      std::string owned = "hello";       // owned string
      owned += " world";                 // mutable
      print(owned);                      // implicit conversion
      print("literal");                  // works with literals
    }

**Rust:**

.. code-block:: rust

    fn print(s: &str) {  // borrowed string slice
        println!("{}", s);
    }

    fn main() {
        let owned = String::from("hello");  // owned String
        let owned = owned + " world";       // concatenation
        print(&owned);                      // borrow as &str
        print("literal");                   // &str literal
    }

Creating Strings
----------------

.. code-block:: rust

    // From literal
    let s1 = String::from("hello");
    let s2 = "hello".to_string();
    let s3: String = "hello".into();

    // Empty string
    let s4 = String::new();

    // With capacity
    let s5 = String::with_capacity(100);

    // From format
    let name = "world";
    let s6 = format!("Hello, {}!", name);

String Operations
-----------------

**Concatenation:**

.. code-block:: rust

    let s1 = String::from("Hello");
    let s2 = String::from(" World");

    // Using + (takes ownership of s1)
    let s3 = s1 + &s2;  // s1 moved, s2 borrowed

    // Using format! (doesn't take ownership)
    let s1 = String::from("Hello");
    let s4 = format!("{}{}", s1, s2);  // s1 and s2 still valid

    // Using push_str
    let mut s5 = String::from("Hello");
    s5.push_str(" World");

**Slicing:**

.. code-block:: rust

    let s = String::from("hello world");

    let hello = &s[0..5];   // "hello"
    let world = &s[6..];    // "world"
    let full = &s[..];      // "hello world"

    // Warning: indices are byte positions, not char positions
    // Slicing in middle of UTF-8 char will panic

**Iteration:**

.. code-block:: rust

    let s = "hello";

    // By characters
    for c in s.chars() {
        println!("{}", c);
    }

    // By bytes
    for b in s.bytes() {
        println!("{}", b);
    }

Common Methods
--------------

.. code-block:: rust

    let s = String::from("  Hello World  ");

    // Trimming
    let trimmed = s.trim();           // "Hello World"
    let left = s.trim_start();        // "Hello World  "
    let right = s.trim_end();         // "  Hello World"

    // Case conversion
    let upper = s.to_uppercase();     // "  HELLO WORLD  "
    let lower = s.to_lowercase();     // "  hello world  "

    // Searching
    let contains = s.contains("World");           // true
    let starts = s.starts_with("  Hello");        // true
    let pos = s.find("World");                    // Some(8)

    // Replacing
    let replaced = s.replace("World", "Rust");    // "  Hello Rust  "

    // Splitting
    let parts: Vec<&str> = "a,b,c".split(',').collect();  // ["a", "b", "c"]

String vs &str in Function Parameters
-------------------------------------

Prefer ``&str`` for function parameters when you don't need ownership:

.. code-block:: rust

    // Good: accepts both String and &str
    fn greet(name: &str) {
        println!("Hello, {}!", name);
    }

    fn main() {
        let owned = String::from("Alice");
        greet(&owned);     // borrow String as &str
        greet("Bob");      // &str literal directly
    }

    // Less flexible: only accepts String
    fn greet_owned(name: String) {
        println!("Hello, {}!", name);
    }

UTF-8 Handling
--------------

Rust strings are always valid UTF-8. This differs from C++ where strings are byte
sequences:

.. code-block:: rust

    let s = "Hello, 世界!";

    // Length in bytes vs characters
    println!("Bytes: {}", s.len());        // 14 bytes
    println!("Chars: {}", s.chars().count()); // 10 characters

    // Accessing by index doesn't work directly
    // let c = s[0];  // error: cannot index String

    // Use chars() or bytes()
    let first_char = s.chars().next();     // Some('H')
    let first_byte = s.bytes().next();     // Some(72)

Converting Between Types
------------------------

.. code-block:: rust

    // &str to String
    let s: &str = "hello";
    let owned: String = s.to_string();
    let owned: String = s.to_owned();
    let owned: String = String::from(s);

    // String to &str
    let owned = String::from("hello");
    let slice: &str = &owned;
    let slice: &str = owned.as_str();

    // Numbers to String
    let n = 42;
    let s = n.to_string();
    let s = format!("{}", n);

    // String to numbers
    let s = "42";
    let n: i32 = s.parse().unwrap();
    let n: i32 = s.parse().expect("not a number");

See Also
--------

- :doc:`rust_basic` - Ownership and borrowing
- :doc:`rust_error` - Error handling with parse()
