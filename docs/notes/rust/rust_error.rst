==============
Error Handling
==============

.. meta::
   :description: Rust error handling with Result and Option compared to C++ exceptions. Covers ? operator, unwrap, expect, and error propagation.
   :keywords: Rust, Result, Option, error handling, C++ exceptions, unwrap, ? operator

.. contents:: Table of Contents
    :backlinks: none

:Source: `src/rust/error_handling <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/error_handling>`_

Rust uses ``Result<T, E>`` and ``Option<T>`` for error handling instead of exceptions.
This approach makes error handling explicit - you can see from a function's signature
whether it can fail and what errors it might return. Unlike exceptions which can be
thrown from anywhere and caught anywhere (or not caught at all), Rust errors are values
that must be explicitly handled at each call site. This eliminates the "invisible control
flow" problem of exceptions and makes code easier to reason about.

Error Handling Comparison
-------------------------

The following table maps C++ error handling mechanisms to their Rust equivalents. Note
that Rust's ``Result`` is similar to C++23's ``std::expected``, but ``Result`` has been
in Rust since 1.0 and is deeply integrated into the language with the ``?`` operator:

+---------------------------+---------------------------+
| C++                       | Rust                      |
+===========================+===========================+
| ``throw exception``       | ``Err(e)``                |
+---------------------------+---------------------------+
| ``try { } catch { }``     | ``match`` / ``?``         |
+---------------------------+---------------------------+
| ``std::optional<T>``      | ``Option<T>``             |
+---------------------------+---------------------------+
| ``std::expected<T, E>``   | ``Result<T, E>``          |
+---------------------------+---------------------------+
| ``nullptr`` check         | ``Option<T>``             |
+---------------------------+---------------------------+

Option Type
-----------

``Option<T>`` represents a value that may or may not exist - it's either ``Some(value)``
or ``None``. This replaces null pointers and eliminates null pointer dereference bugs
at compile time. Unlike C++ where you might forget to check if an ``optional`` has a
value before accessing it, Rust forces you to handle both cases through pattern matching
or methods like ``unwrap_or``. The compiler won't let you access the inner value without
acknowledging that it might not exist:

**C++:**

.. code-block:: cpp

    #include <optional>

    std::optional<int> find_index(const std::vector<int>& v, int target) {
      for (size_t i = 0; i < v.size(); i++) {
        if (v[i] == target) return i;
      }
      return std::nullopt;
    }

    int main() {
      std::vector<int> v = {1, 2, 3};
      if (auto idx = find_index(v, 2)) {
        std::cout << "Found at " << *idx;
      }
    }

**Rust:**

.. code-block:: rust

    fn find_index(v: &[i32], target: i32) -> Option<usize> {
        for (i, &x) in v.iter().enumerate() {
            if x == target {
                return Some(i);
            }
        }
        None
    }

    fn main() {
        let v = vec![1, 2, 3];
        if let Some(idx) = find_index(&v, 2) {
            println!("Found at {}", idx);
        }
    }

Option Methods
~~~~~~~~~~~~~~

``Option`` provides many methods for working with optional values without verbose
pattern matching. These methods form a fluent API for transforming, combining, and
extracting optional values. The ``unwrap`` family of methods extracts the value but
panics if it's ``None`` - use these only when you're certain the value exists or in
prototyping code:

.. code-block:: rust

    let x: Option<i32> = Some(42);
    let y: Option<i32> = None;

    // Unwrapping
    let val = x.unwrap();           // panics if None
    let val = x.expect("no value"); // panics with message
    let val = x.unwrap_or(0);       // default if None
    let val = x.unwrap_or_default(); // Default::default() if None
    let val = x.unwrap_or_else(|| compute_default());

    // Checking
    let is_some = x.is_some();
    let is_none = y.is_none();

    // Transforming
    let doubled = x.map(|v| v * 2);           // Some(84)
    let filtered = x.filter(|&v| v > 50);     // None
    let chained = x.and_then(|v| Some(v + 1)); // Some(43)

    // Converting to Result
    let result: Result<i32, &str> = x.ok_or("no value");

Result Type
-----------

``Result<T, E>`` represents an operation that can succeed with a value of type ``T``
or fail with an error of type ``E``. Unlike exceptions which separate the error path
from the normal code path, ``Result`` keeps them together - you handle the error right
where it occurs. This makes error handling more predictable and prevents errors from
silently propagating up the call stack. The type system ensures you can't accidentally
ignore an error:

**C++:**

.. code-block:: cpp

    #include <expected>  // C++23
    #include <string>

    std::expected<int, std::string> parse_int(const std::string& s) {
      try {
        return std::stoi(s);
      } catch (...) {
        return std::unexpected("parse error");
      }
    }

**Rust:**

.. code-block:: rust

    fn parse_int(s: &str) -> Result<i32, std::num::ParseIntError> {
        s.parse()
    }

    fn main() {
        match parse_int("42") {
            Ok(n) => println!("Parsed: {}", n),
            Err(e) => println!("Error: {}", e),
        }
    }

Result Methods
~~~~~~~~~~~~~~

``Result`` provides methods similar to ``Option`` for working with fallible operations.
The ``map`` and ``map_err`` methods let you transform the success or error value
independently. The ``and_then`` method (also called ``flatmap`` in other languages)
chains operations that might fail:

.. code-block:: rust

    let x: Result<i32, &str> = Ok(42);
    let y: Result<i32, &str> = Err("error");

    // Unwrapping
    let val = x.unwrap();           // panics if Err
    let val = x.expect("failed");   // panics with message
    let val = x.unwrap_or(0);       // default if Err
    let val = x.unwrap_or_else(|e| handle_error(e));

    // Checking
    let is_ok = x.is_ok();
    let is_err = y.is_err();

    // Transforming
    let doubled = x.map(|v| v * 2);           // Ok(84)
    let mapped_err = y.map_err(|e| format!("Error: {}", e));

    // Converting to Option
    let opt: Option<i32> = x.ok();   // Some(42)
    let opt: Option<&str> = y.err(); // Some("error")

The ? Operator
--------------

The ``?`` operator is Rust's way of propagating errors concisely. When applied to a
``Result``, it either unwraps the ``Ok`` value and continues, or returns early with
the ``Err`` value. This eliminates the boilerplate of matching on every ``Result``
and manually returning errors. The ``?`` operator also performs automatic error type
conversion using the ``From`` trait, making it easy to combine different error types:

**C++ (manual propagation):**

.. code-block:: cpp

    std::expected<int, Error> read_config() {
      auto file = open_file("config.txt");
      if (!file) return std::unexpected(file.error());

      auto content = read_content(*file);
      if (!content) return std::unexpected(content.error());

      auto value = parse_int(*content);
      if (!value) return std::unexpected(value.error());

      return *value;
    }

**Rust (with ?):**

.. code-block:: rust

    fn read_config() -> Result<i32, Error> {
        let file = open_file("config.txt")?;
        let content = read_content(&file)?;
        let value = parse_int(&content)?;
        Ok(value)
    }

    // ? works with Option too
    fn first_char(s: &str) -> Option<char> {
        let c = s.chars().next()?;
        Some(c)
    }

Custom Error Types
------------------

For libraries and larger applications, you'll want to define custom error types that
can represent all the ways your code can fail. A common pattern is an enum with variants
for each error kind. Implementing ``From`` for other error types enables automatic
conversion with the ``?`` operator, so you can use ``?`` on standard library errors
and have them automatically converted to your custom type:

.. code-block:: rust

    use std::fmt;

    #[derive(Debug)]
    enum AppError {
        IoError(std::io::Error),
        ParseError(std::num::ParseIntError),
        Custom(String),
    }

    impl fmt::Display for AppError {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match self {
                AppError::IoError(e) => write!(f, "IO error: {}", e),
                AppError::ParseError(e) => write!(f, "Parse error: {}", e),
                AppError::Custom(s) => write!(f, "{}", s),
            }
        }
    }

    impl std::error::Error for AppError {}

    // Implement From for automatic conversion with ?
    impl From<std::io::Error> for AppError {
        fn from(e: std::io::Error) -> Self {
            AppError::IoError(e)
        }
    }

Pattern Matching Errors
-----------------------

Pattern matching gives you fine-grained control over error handling. You can match
on specific error variants, use guards to add conditions, and handle multiple cases
differently. The ``if let`` syntax is convenient when you only care about one case,
and ``let else`` (Rust 1.65+) provides early return for the error case:

.. code-block:: rust

    fn process(input: &str) -> Result<i32, String> {
        match input.parse::<i32>() {
            Ok(n) if n > 0 => Ok(n * 2),
            Ok(_) => Err("must be positive".into()),
            Err(e) => Err(format!("parse error: {}", e)),
        }
    }

    // if let for single case
    if let Ok(n) = "42".parse::<i32>() {
        println!("{}", n);
    }

    // let else for early return
    fn process2(input: &str) -> i32 {
        let Ok(n) = input.parse::<i32>() else {
            return 0;
        };
        n * 2
    }

Panic vs Result
---------------

Rust distinguishes between recoverable errors (``Result``) and unrecoverable errors
(``panic!``). Use ``panic!`` for bugs and invariant violations - situations that
indicate a programming error rather than an expected failure. Use ``Result`` for
operations that can legitimately fail, like file I/O or parsing user input. The
``assert!`` macro panics in debug builds but can be compiled out in release builds
for performance-critical code:

.. code-block:: rust

    // panic! - unrecoverable errors (bugs, invariant violations)
    fn get_index(v: &[i32], i: usize) -> i32 {
        if i >= v.len() {
            panic!("index out of bounds");
        }
        v[i]
    }

    // Result - recoverable errors (expected failures)
    fn read_file(path: &str) -> Result<String, std::io::Error> {
        std::fs::read_to_string(path)
    }

    // assert! - debug checks (removed in release)
    fn divide(a: i32, b: i32) -> i32 {
        debug_assert!(b != 0, "division by zero");
        a / b
    }

See Also
--------

- :doc:`rust_traits` - Error trait
- :doc:`rust_iterator` - Iterator methods returning Option/Result
