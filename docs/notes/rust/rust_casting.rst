=======
Casting
=======

.. meta::
   :description: Rust type casting and conversions compared to C++. Covers as keyword, From/Into traits, TryFrom/TryInto for fallible conversions.
   :keywords: Rust, casting, as, From, Into, TryFrom, TryInto, type conversion, C++ static_cast

.. contents:: Table of Contents
    :backlinks: none

:Source: `src/rust/casting <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/casting>`_

Rust provides several mechanisms for type conversion, each with different safety
guarantees. Unlike C++, Rust has no implicit numeric conversions - every conversion
must be explicit. This prevents subtle bugs from unintended narrowing or widening
conversions. Rust's type conversion system is built around traits (``From``, ``Into``,
``TryFrom``, ``TryInto``) that provide a consistent interface for both infallible
and fallible conversions. The ``as`` keyword is reserved for primitive type casts
where the conversion rules are well-defined by the language.

Casting Comparison
------------------

The following table maps C++ casting mechanisms to their Rust equivalents. Note that
Rust deliberately omits some C++ casts - ``dynamic_cast`` isn't needed because Rust
uses trait objects with explicit ``dyn`` syntax, and ``reinterpret_cast`` requires
``unsafe`` code in Rust:

+------------------------+---------------------------+
| C++                    | Rust                      |
+========================+===========================+
| ``static_cast<T>``     | ``as T`` (primitives)     |
+------------------------+---------------------------+
| Conversion constructor | ``From<T>`` / ``Into<T>`` |
+------------------------+---------------------------+
| ``explicit`` conversion| ``TryFrom<T>``            |
+------------------------+---------------------------+
| ``dynamic_cast<T>``    | No direct equivalent      |
+------------------------+---------------------------+
| ``reinterpret_cast<T>``| ``transmute`` (unsafe)    |
+------------------------+---------------------------+

Primitive Casting with ``as``
-----------------------------

The ``as`` keyword performs primitive type conversions similar to C++'s
``static_cast``. However, unlike C++, Rust never performs implicit numeric
conversions - you must always use ``as`` explicitly. This example shows basic
numeric conversions:

**C++:**

.. code-block:: cpp

    #include <iostream>

    int main() {
      int i = 42;

      // Explicit casts
      double d = static_cast<double>(i);
      char c = static_cast<char>(i);

      // Implicit conversions (allowed in C++, can be surprising)
      double d2 = i;           // implicit widening
      short s = i;             // implicit narrowing (may warn)
      unsigned int u = -1;     // implicit sign conversion (dangerous!)

      std::cout << "d: " << d << ", c: " << static_cast<int>(c) << "\n";
      std::cout << "d2: " << d2 << ", s: " << s << ", u: " << u << "\n";

      return 0;
    }

**Rust:**

.. code-block:: rust

    fn main() {
        let i: i32 = 42;

        // Explicit casts required
        let d: f64 = i as f64;
        let c: u8 = i as u8;

        // No implicit conversions - these won't compile:
        // let d2: f64 = i;        // error: expected f64, found i32
        // let s: i16 = i;         // error: expected i16, found i32
        // let u: u32 = -1_i32;    // error: expected u32, found i32

        // Must be explicit about everything
        let d2: f64 = i as f64;
        let s: i16 = i as i16;
        let u: u32 = (-1_i32) as u32;  // wraps to u32::MAX

        println!("d: {}, c: {}", d, c);
        println!("d2: {}, s: {}, u: {}", d2, s, u);
    }

``as`` Casting Rules
~~~~~~~~~~~~~~~~~~~~

The ``as`` keyword follows specific rules for different conversion types. Numeric
conversions may truncate or wrap, float-to-int truncates toward zero, and pointer
casts are allowed between compatible types. This example demonstrates the various
``as`` casting behaviors:

**C++:**

.. code-block:: cpp

    #include <cstdint>
    #include <iostream>

    int main() {
      // Narrowing: truncates high bits
      int32_t x = 1000;
      uint8_t y = static_cast<uint8_t>(x);  // 232 (1000 % 256)

      // Float to int: truncates toward zero
      double f = 3.9;
      int i = static_cast<int>(f);  // 3

      double neg = -3.9;
      int neg_i = static_cast<int>(neg);  // -3

      // Pointer to integer
      int value = 42;
      int* ptr = &value;
      uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);

      std::cout << "y: " << static_cast<int>(y) << "\n";
      std::cout << "i: " << i << ", neg_i: " << neg_i << "\n";
      std::cout << "addr: " << addr << "\n";

      return 0;
    }

**Rust:**

.. code-block:: rust

    fn main() {
        // Narrowing: truncates high bits (wrapping behavior)
        let x: i32 = 1000;
        let y: u8 = x as u8;  // 232 (1000 % 256)
        println!("1000 as u8 = {}", y);

        // Float to int: truncates toward zero
        let f: f64 = 3.9;
        let i: i32 = f as i32;  // 3
        println!("3.9 as i32 = {}", i);

        let neg: f64 = -3.9;
        let neg_i: i32 = neg as i32;  // -3
        println!("-3.9 as i32 = {}", neg_i);

        // Pointer casts
        let value: i32 = 42;
        let ptr: *const i32 = &value;
        let addr: usize = ptr as usize;
        println!("Address: {:#x}", addr);

        // Reference to raw pointer
        let r: &i32 = &value;
        let raw: *const i32 = r as *const i32;
        println!("Raw pointer: {:?}", raw);
    }

From and Into Traits
--------------------

``From`` and ``Into`` are traits for infallible type conversions. They're the
idiomatic way to convert between types in Rust, similar to C++ conversion
constructors but more explicit and composable. Implementing ``From<T>`` for a type
automatically provides the ``Into<T>`` implementation for free. These traits are
used extensively in the standard library and are the preferred way to handle
type conversions:

**C++ (conversion constructor):**

.. code-block:: cpp

    #include <iostream>

    class Fahrenheit;  // forward declaration

    class Celsius {
    public:
      double value;

      explicit Celsius(double v) : value(v) {}

      // Conversion constructor from Fahrenheit
      explicit Celsius(const Fahrenheit& f);

      void print() const {
        std::cout << value << "°C";
      }
    };

    class Fahrenheit {
    public:
      double value;

      explicit Fahrenheit(double v) : value(v) {}

      // Conversion constructor from Celsius
      explicit Fahrenheit(const Celsius& c)
        : value(c.value * 9.0 / 5.0 + 32.0) {}

      void print() const {
        std::cout << value << "°F";
      }
    };

    // Define after Fahrenheit is complete
    Celsius::Celsius(const Fahrenheit& f)
      : value((f.value - 32.0) * 5.0 / 9.0) {}

    int main() {
      Fahrenheit body_temp(98.6);
      Celsius c(body_temp);  // explicit conversion

      std::cout << "Body temperature: ";
      body_temp.print();
      std::cout << " = ";
      c.print();
      std::cout << "\n";

      return 0;
    }

**Rust:**

.. code-block:: rust

    #[derive(Debug)]
    struct Celsius(f64);

    #[derive(Debug)]
    struct Fahrenheit(f64);

    // Implement From<Fahrenheit> for Celsius
    impl From<Fahrenheit> for Celsius {
        fn from(f: Fahrenheit) -> Self {
            Celsius((f.0 - 32.0) * 5.0 / 9.0)
        }
    }

    // Implement From<Celsius> for Fahrenheit
    impl From<Celsius> for Fahrenheit {
        fn from(c: Celsius) -> Self {
            Fahrenheit(c.0 * 9.0 / 5.0 + 32.0)
        }
    }

    fn main() {
        let body_temp = Fahrenheit(98.6);

        // Using From::from explicitly
        let c1 = Celsius::from(Fahrenheit(98.6));
        println!("Body temperature: {:?} = {:?}", body_temp, c1);

        // Using Into (automatically available when From is implemented)
        let f2 = Fahrenheit(212.0);
        let c2: Celsius = f2.into();
        println!("Boiling point: 212°F = {:?}", c2);

        // Into is useful in generic contexts
        fn print_celsius(temp: impl Into<Celsius>) {
            let c = temp.into();
            println!("Temperature: {:.1}°C", c.0);
        }

        print_celsius(Fahrenheit(32.0));  // accepts Fahrenheit
        print_celsius(Celsius(0.0));      // also accepts Celsius (From<T> for T)
    }

Common From Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The standard library provides many ``From`` implementations for common conversions.
These make it easy to convert between related types without explicit casting:

**C++:**

.. code-block:: cpp

    #include <filesystem>
    #include <iostream>
    #include <string>
    #include <vector>

    int main() {
      // String from literal
      std::string s1 = "hello";  // implicit conversion
      std::string s2("hello");   // explicit construction

      // Vector from initializer list
      std::vector<int> v = {1, 2, 3};

      // Path from string
      std::filesystem::path p = "/tmp/file.txt";

      std::cout << "String: " << s1 << "\n";
      std::cout << "Vector size: " << v.size() << "\n";
      std::cout << "Path: " << p << "\n";

      return 0;
    }

**Rust:**

.. code-block:: rust

    use std::path::PathBuf;

    fn main() {
        // String from &str
        let s1: String = String::from("hello");
        let s2: String = "hello".into();  // equivalent
        let s3: String = "hello".to_string();  // also common
        println!("Strings: {}, {}, {}", s1, s2, s3);

        // Vec from array
        let v1: Vec<i32> = Vec::from([1, 2, 3]);
        let v2: Vec<i32> = [1, 2, 3].into();
        println!("Vectors: {:?}, {:?}", v1, v2);

        // PathBuf from string
        let p1: PathBuf = PathBuf::from("/tmp/file.txt");
        let p2: PathBuf = "/tmp/file.txt".into();
        println!("Paths: {:?}, {:?}", p1, p2);

        // Box from value
        let boxed: Box<i32> = Box::from(42);
        let boxed2: Box<i32> = 42.into();
        println!("Boxed: {}, {}", boxed, boxed2);

        // String from number (via ToString trait, not From)
        let num_str = 42.to_string();
        println!("Number as string: {}", num_str);
    }

TryFrom and TryInto
-------------------

For conversions that might fail, Rust provides ``TryFrom`` and ``TryInto`` traits.
These return ``Result<T, E>`` instead of the converted value directly, forcing
callers to handle potential errors. This is similar to C++ functions that return
``std::optional`` or throw exceptions:

**C++:**

.. code-block:: cpp

    #include <cstdint>
    #include <iostream>
    #include <optional>
    #include <stdexcept>

    // Return optional for fallible conversion
    std::optional<uint8_t> try_from_int(int value) {
      if (value < 0 || value > 255) {
        return std::nullopt;
      }
      return static_cast<uint8_t>(value);
    }

    // Or throw exception
    uint8_t from_int_throwing(int value) {
      if (value < 0 || value > 255) {
        throw std::out_of_range("Value out of u8 range");
      }
      return static_cast<uint8_t>(value);
    }

    int main() {
      // Using optional
      if (auto result = try_from_int(100)) {
        std::cout << "Converted: " << static_cast<int>(*result) << "\n";
      }

      if (auto result = try_from_int(1000)) {
        std::cout << "This won't print\n";
      } else {
        std::cout << "Conversion failed for 1000\n";
      }

      // Using exception
      try {
        auto val = from_int_throwing(1000);
      } catch (const std::out_of_range& e) {
        std::cout << "Exception: " << e.what() << "\n";
      }

      return 0;
    }

**Rust:**

.. code-block:: rust

    use std::convert::TryFrom;

    fn main() {
        let big: i32 = 1000;

        // TryFrom returns Result - must handle error
        let result: Result<u8, _> = u8::try_from(big);
        match result {
            Ok(val) => println!("Converted: {}", val),
            Err(e) => println!("Conversion failed: {}", e),
        }

        // Using ? operator in functions that return Result
        let fits: i32 = 100;
        let small: u8 = u8::try_from(fits).expect("Value should fit");
        println!("Converted 100 to u8: {}", small);

        // TryInto is also available
        let big2: i32 = 1000;
        let result2: Result<u8, _> = big2.try_into();
        println!("TryInto result: {:?}", result2);
    }

Custom TryFrom
~~~~~~~~~~~~~~

You can implement ``TryFrom`` for your own types to provide fallible conversions
with custom error types. This is useful for validation during construction:

**C++:**

.. code-block:: cpp

    #include <iostream>
    #include <optional>
    #include <stdexcept>
    #include <string>

    class PositiveInt {
      int value_;

    public:
      // Factory function returning optional
      static std::optional<PositiveInt> try_from(int value) {
        if (value > 0) {
          return PositiveInt(value);
        }
        return std::nullopt;
      }

      // Or throwing constructor
      explicit PositiveInt(int value) : value_(value) {
        if (value <= 0) {
          throw std::invalid_argument("value must be positive");
        }
      }

      int get() const { return value_; }
    };

    int main() {
      // Using optional factory
      if (auto pos = PositiveInt::try_from(42)) {
        std::cout << "Created: " << pos->get() << "\n";
      }

      if (auto neg = PositiveInt::try_from(-1)) {
        std::cout << "This won't print\n";
      } else {
        std::cout << "Failed to create from -1\n";
      }

      return 0;
    }

**Rust:**

.. code-block:: rust

    use std::convert::TryFrom;
    use std::fmt;

    #[derive(Debug)]
    struct PositiveInt(i32);

    #[derive(Debug)]
    struct NotPositiveError(i32);

    impl fmt::Display for NotPositiveError {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "{} is not positive", self.0)
        }
    }

    impl TryFrom<i32> for PositiveInt {
        type Error = NotPositiveError;

        fn try_from(value: i32) -> Result<Self, Self::Error> {
            if value > 0 {
                Ok(PositiveInt(value))
            } else {
                Err(NotPositiveError(value))
            }
        }
    }

    fn main() {
        // Successful conversion
        let pos = PositiveInt::try_from(42);
        println!("From 42: {:?}", pos);  // Ok(PositiveInt(42))

        // Failed conversion
        let neg = PositiveInt::try_from(-1);
        println!("From -1: {:?}", neg);  // Err(NotPositiveError(-1))

        // Pattern matching on result
        match PositiveInt::try_from(100) {
            Ok(p) => println!("Created positive int: {:?}", p),
            Err(e) => println!("Error: {}", e),
        }
    }

String Conversions
------------------

String parsing and formatting are common conversion operations. Rust uses the
``FromStr`` trait (called via ``.parse()``) for parsing and ``ToString``/``Display``
for formatting:

**C++:**

.. code-block:: cpp

    #include <iostream>
    #include <sstream>
    #include <string>

    int main() {
      // Number to string
      int n = 42;
      std::string s1 = std::to_string(n);
      std::cout << "to_string: " << s1 << "\n";

      // Using stringstream for formatting
      std::ostringstream oss;
      oss << "Value: " << n;
      std::string s2 = oss.str();
      std::cout << "stringstream: " << s2 << "\n";

      // String to number
      std::string num_str = "42";
      int parsed = std::stoi(num_str);
      std::cout << "Parsed: " << parsed << "\n";

      // Error handling
      try {
        int bad = std::stoi("not a number");
      } catch (const std::invalid_argument& e) {
        std::cout << "Parse error: " << e.what() << "\n";
      }

      return 0;
    }

**Rust:**

.. code-block:: rust

    use std::str::FromStr;

    fn main() {
        // Number to String
        let n = 42;
        let s1: String = n.to_string();
        let s2: String = format!("Value: {}", n);
        println!("to_string: {}", s1);
        println!("format!: {}", s2);

        // String to number using parse()
        let num_str = "42";
        let parsed: i32 = num_str.parse().unwrap();
        println!("Parsed: {}", parsed);

        // With turbofish syntax
        let parsed2 = "42".parse::<i32>().unwrap();
        println!("Turbofish: {}", parsed2);

        // Using FromStr directly
        let parsed3 = i32::from_str("42").unwrap();
        println!("FromStr: {}", parsed3);

        // Error handling - parse returns Result
        let bad: Result<i32, _> = "not a number".parse();
        match bad {
            Ok(n) => println!("Parsed: {}", n),
            Err(e) => println!("Parse error: {}", e),
        }
    }

Deref Coercion
--------------

Rust automatically applies deref coercion when a type implements the ``Deref``
trait. This allows ``&String`` to be used where ``&str`` is expected, and
``&Box<T>`` where ``&T`` is expected. This is similar to C++'s implicit
conversions but more controlled:

**C++:**

.. code-block:: cpp

    #include <iostream>
    #include <memory>
    #include <string>

    void print_cstr(const char* s) {
      std::cout << s << "\n";
    }

    void print_int_ref(const int& n) {
      std::cout << n << "\n";
    }

    int main() {
      std::string s = "hello";

      // std::string implicitly converts to const char*
      print_cstr(s.c_str());  // explicit .c_str() needed

      // unique_ptr dereferences to T
      auto ptr = std::make_unique<int>(42);
      print_int_ref(*ptr);  // explicit * needed

      return 0;
    }

**Rust:**

.. code-block:: rust

    fn print_str(s: &str) {
        println!("{}", s);
    }

    fn print_int_ref(n: &i32) {
        println!("{}", n);
    }

    fn main() {
        let s = String::from("hello");

        // String automatically derefs to &str - no explicit conversion!
        print_str(&s);

        // Box<T> automatically derefs to &T
        let boxed = Box::new(42);
        print_int_ref(&boxed);  // auto-deref, no * needed

        // Works through multiple levels
        let boxed_string = Box::new(String::from("nested"));
        print_str(&boxed_string);  // Box<String> -> String -> &str

        // Vec<T> derefs to &[T]
        fn print_slice(s: &[i32]) {
            println!("{:?}", s);
        }
        let v = vec![1, 2, 3];
        print_slice(&v);  // Vec<i32> -> &[i32]
    }

AsRef and AsMut
---------------

``AsRef`` and ``AsMut`` traits provide cheap reference-to-reference conversions.
They're commonly used in generic functions to accept multiple types that can be
viewed as a reference to some target type:

**C++:**

.. code-block:: cpp

    #include <filesystem>
    #include <iostream>
    #include <string>
    #include <string_view>

    // Overloads for different string types
    void print_path(const std::filesystem::path& p) {
      std::cout << p << "\n";
    }

    void print_path(const std::string& s) {
      print_path(std::filesystem::path(s));
    }

    void print_path(const char* s) {
      print_path(std::filesystem::path(s));
    }

    int main() {
      print_path("/tmp/file.txt");
      print_path(std::string("/home/user"));
      print_path(std::filesystem::path("/var/log"));

      return 0;
    }

**Rust:**

.. code-block:: rust

    use std::path::Path;

    // Single generic function accepts anything that can be viewed as &Path
    fn print_path<P: AsRef<Path>>(path: P) {
        println!("{}", path.as_ref().display());
    }

    fn main() {
        // All of these work with the same function!
        print_path("hello.txt");                    // &str
        print_path(String::from("hi.txt"));         // String
        print_path(Path::new("x.txt"));             // &Path
        print_path(std::path::PathBuf::from("y"));  // PathBuf

        // AsRef is also useful for byte slices
        fn process_bytes<T: AsRef<[u8]>>(data: T) {
            let bytes = data.as_ref();
            println!("Got {} bytes", bytes.len());
        }

        process_bytes("hello");           // &str -> &[u8]
        process_bytes(vec![1, 2, 3]);     // Vec<u8> -> &[u8]
        process_bytes(&[4, 5, 6][..]);    // &[u8]
    }

Unsafe Transmute
----------------

``std::mem::transmute`` reinterprets the bits of one type as another, similar to
C++'s ``reinterpret_cast``. This is unsafe and should only be used when you're
certain the bit patterns are valid for both types:

**C++:**

.. code-block:: cpp

    #include <cstdint>
    #include <cstring>
    #include <iostream>

    int main() {
      // reinterpret_cast for pointer types
      uint32_t x = 0x12345678;
      uint8_t* bytes = reinterpret_cast<uint8_t*>(&x);

      std::cout << "Bytes: ";
      for (int i = 0; i < 4; ++i) {
        std::cout << std::hex << static_cast<int>(bytes[i]) << " ";
      }
      std::cout << "\n";

      // memcpy for type punning (safer than reinterpret_cast)
      float f = 3.14f;
      uint32_t bits;
      std::memcpy(&bits, &f, sizeof(f));
      std::cout << "Float bits: 0x" << std::hex << bits << "\n";

      return 0;
    }

**Rust:**

.. code-block:: rust

    fn main() {
        // UNSAFE: transmute reinterprets bits
        unsafe {
            let x: u32 = 0x12345678;
            let bytes: [u8; 4] = std::mem::transmute(x);
            println!("Bytes: {:02x?}", bytes);

            // Float to bits
            let f: f32 = 3.14;
            let bits: u32 = std::mem::transmute(f);
            println!("Float bits: {:#x}", bits);
        }

        // Safe alternatives exist for common cases:
        let x: u32 = 0x12345678;
        let bytes = x.to_ne_bytes();  // native endian
        println!("Safe bytes: {:02x?}", bytes);

        let f: f32 = 3.14;
        let bits = f.to_bits();  // safe float-to-bits
        println!("Safe float bits: {:#x}", bits);
    }

See Also
--------

- :doc:`rust_traits` - From/Into as traits
- :doc:`rust_error` - TryFrom returns Result
