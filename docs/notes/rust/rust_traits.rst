===================
Traits and Generics
===================

.. meta::
   :description: Rust traits and generics compared to C++ templates and concepts. Covers trait bounds, impl blocks, generic functions, and associated types.
   :keywords: Rust, traits, generics, C++ templates, concepts, trait bounds, impl, where clause

.. contents:: Table of Contents
    :backlinks: none

:Source: `src/rust/generics <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/generics>`_

Rust traits are the foundation of Rust's approach to polymorphism and code reuse.
They define shared behavior that types can implement, similar to interfaces in other
languages. Unlike C++ templates which use duck typing (any type that has the required
operations will work), Rust generics require explicit trait bounds that are checked
at compile time. This means you get clear error messages at the call site rather than
deep inside template instantiation. Traits also enable Rust's powerful derive system,
operator overloading, and the orphan rules that prevent conflicting implementations.

Traits vs C++ Concepts/Interfaces
---------------------------------

Traits serve a similar purpose to C++ abstract classes and C++20 concepts, but with
key differences. Unlike abstract classes, traits don't require inheritance and can
be implemented for any type, including primitives and types from other crates (with
restrictions). Unlike C++ concepts which are constraints on templates, Rust traits
are first-class types that can be used for both static and dynamic dispatch.

The following example shows how to define a trait and implement it for a custom type.
Note that Rust separates data (struct) from behavior (impl), unlike C++ where methods
are typically defined inside the class:

**C++ (abstract class):**

.. code-block:: cpp

    #include <iostream>

    // Abstract base class defines interface
    class Printable {
    public:
      virtual void print() const = 0;
      virtual ~Printable() = default;
    };

    // Concrete class inherits and implements
    class Point : public Printable {
      int x_, y_;
    public:
      Point(int x, int y) : x_(x), y_(y) {}

      void print() const override {
        std::cout << "(" << x_ << ", " << y_ << ")";
      }

      // Additional methods specific to Point
      int x() const { return x_; }
      int y() const { return y_; }
    };

    // Function accepting any Printable
    void display(const Printable& p) {
      p.print();
      std::cout << "\n";
    }

    int main() {
      Point p(3, 4);
      display(p);  // (3, 4)
      return 0;
    }

**Rust:**

.. code-block:: rust

    // Trait defines shared behavior
    trait Printable {
        fn print(&self);
    }

    // Struct defines data
    struct Point {
        x: i32,
        y: i32,
    }

    // impl block implements trait for type
    impl Printable for Point {
        fn print(&self) {
            println!("({}, {})", self.x, self.y);
        }
    }

    // Additional methods in separate impl block
    impl Point {
        fn new(x: i32, y: i32) -> Self {
            Point { x, y }
        }
    }

    // Function accepting any Printable (static dispatch)
    fn display(p: &impl Printable) {
        p.print();
    }

    fn main() {
        let p = Point::new(3, 4);
        display(&p);  // (3, 4)
    }

Default Implementations
~~~~~~~~~~~~~~~~~~~~~~~

Traits can provide default implementations for methods. Types implementing the trait
can use the default or override it. This is similar to non-pure virtual methods in
C++ abstract classes:

**C++:**

.. code-block:: cpp

    #include <iostream>
    #include <string>

    class Greet {
    public:
      virtual std::string name() const = 0;

      // Default implementation using pure virtual method
      virtual void greet() const {
        std::cout << "Hello, " << name() << "!\n";
      }

      virtual ~Greet() = default;
    };

    class Person : public Greet {
      std::string name_;
    public:
      Person(const std::string& name) : name_(name) {}

      std::string name() const override { return name_; }
      // greet() uses default implementation
    };

    int main() {
      Person p("Alice");
      p.greet();  // Hello, Alice!
      return 0;
    }

**Rust:**

.. code-block:: rust

    trait Greet {
        // Required method - must be implemented
        fn name(&self) -> &str;

        // Default implementation - can be overridden
        fn greet(&self) {
            println!("Hello, {}!", self.name());
        }

        // Another default that calls other trait methods
        fn formal_greet(&self) {
            println!("Good day, {}. How do you do?", self.name());
        }
    }

    struct Person {
        name: String,
    }

    impl Greet for Person {
        fn name(&self) -> &str {
            &self.name
        }
        // greet() and formal_greet() use default implementations
    }

    struct Robot {
        id: u32,
    }

    impl Greet for Robot {
        fn name(&self) -> &str {
            "Robot"
        }

        // Override default implementation
        fn greet(&self) {
            println!("BEEP BOOP. Unit {} operational.", self.id);
        }
    }

    fn main() {
        let person = Person { name: String::from("Alice") };
        person.greet();        // Hello, Alice!
        person.formal_greet(); // Good day, Alice. How do you do?

        let robot = Robot { id: 42 };
        robot.greet();         // BEEP BOOP. Unit 42 operational.
        robot.formal_greet();  // Good day, Robot. How do you do?
    }

Generic Functions
-----------------

Generic functions work with any type that satisfies the specified trait bounds.
Unlike C++ templates where constraints are implicit (duck typing), Rust requires
explicit bounds that are checked at the function definition site:

**C++ template:**

.. code-block:: cpp

    #include <iostream>

    // Unconstrained template - any type with operator> works
    template<typename T>
    T max_value(T a, T b) {
      return (a > b) ? a : b;
    }

    // With concepts (C++20) - explicit constraint
    template<typename T>
    concept Comparable = requires(T a, T b) {
      { a > b } -> std::convertible_to<bool>;
    };

    template<Comparable T>
    T max_concept(T a, T b) {
      return (a > b) ? a : b;
    }

**Rust:**

.. code-block:: rust

    // Trait bound required
    fn max<T: PartialOrd>(a: T, b: T) -> T {
        if a > b { a } else { b }
    }

    // Alternative: where clause
    fn max_where<T>(a: T, b: T) -> T
    where
        T: PartialOrd,
    {
        if a > b { a } else { b }
    }

    // impl Trait syntax (simpler for single use)
    fn max_impl(a: impl PartialOrd, b: impl PartialOrd) -> impl PartialOrd {
        // Note: a and b must be same type in practice
        if a > b { a } else { b }
    }

Multiple Trait Bounds
~~~~~~~~~~~~~~~~~~~~~

When a generic type needs to satisfy multiple constraints, you can combine trait
bounds with ``+``. For complex bounds, the ``where`` clause provides better
readability:

**C++:**

.. code-block:: cpp

    #include <concepts>
    #include <iostream>

    // Multiple constraints with concepts (C++20)
    template<typename T>
    concept DebugPrintable = requires(T t) {
      { std::cout << t };  // can be printed
    };

    template<typename T>
    concept Cloneable = std::copyable<T>;

    // Combining concepts
    template<typename T>
    requires DebugPrintable<T> && Cloneable<T>
    void process(T value) {
      T copy = value;
      std::cout << "Processing: " << copy << "\n";
    }

    int main() {
      process(42);
      process(std::string("hello"));
      return 0;
    }

**Rust:**

.. code-block:: rust

    use std::fmt::{Debug, Display};

    // Multiple bounds with + syntax
    fn print_both<T: Debug + Display>(value: T) {
        println!("Debug: {:?}", value);
        println!("Display: {}", value);
    }

    // Where clause for complex bounds (more readable)
    fn complex_function<T, U>(t: T, u: U) -> String
    where
        T: Debug + Clone + Default,
        U: Display + PartialOrd + Into<String>,
    {
        let t_clone = t.clone();
        println!("Debug of t: {:?}", t_clone);
        u.into()
    }

    fn main() {
        // i32 implements both Debug and Display
        print_both(42);

        // String also implements both
        print_both(String::from("hello"));
    }

Generic Structs
---------------

Generic structs allow you to define data structures that work with any type. You
can add trait bounds to restrict which types can be used, and you can provide
specialized implementations for specific types:

**C++:**

.. code-block:: cpp

    #include <iostream>

    template<typename T>
    struct Pair {
      T first;
      T second;

      Pair(T f, T s) : first(f), second(s) {}

      void print() const {
        std::cout << "(" << first << ", " << second << ")\n";
      }
    };

    // Specialization for int
    template<>
    struct Pair<int> {
      int first;
      int second;

      Pair(int f, int s) : first(f), second(s) {}

      int sum() const { return first + second; }

      void print() const {
        std::cout << "(" << first << ", " << second << ") sum=" << sum() << "\n";
      }
    };

    int main() {
      Pair<double> pd(1.5, 2.5);
      pd.print();  // (1.5, 2.5)

      Pair<int> pi(3, 4);
      pi.print();  // (3, 4) sum=7
      std::cout << "Sum: " << pi.sum() << "\n";

      return 0;
    }

**Rust:**

.. code-block:: rust

    use std::fmt::Display;

    // Generic struct
    struct Pair<T> {
        first: T,
        second: T,
    }

    // Methods for all Pair<T>
    impl<T> Pair<T> {
        fn new(first: T, second: T) -> Self {
            Pair { first, second }
        }
    }

    // Methods only when T: Display
    impl<T: Display> Pair<T> {
        fn print(&self) {
            println!("({}, {})", self.first, self.second);
        }
    }

    // Methods only for Pair<i32> (specialization)
    impl Pair<i32> {
        fn sum(&self) -> i32 {
            self.first + self.second
        }
    }

    // Methods only when T supports addition
    impl<T: std::ops::Add<Output = T> + Copy> Pair<T> {
        fn add(&self) -> T {
            self.first + self.second
        }
    }

    fn main() {
        let pd = Pair::new(1.5, 2.5);
        pd.print();  // (1.5, 2.5)
        println!("Add: {}", pd.add());  // 4.0

        let pi = Pair::new(3, 4);
        pi.print();  // (3, 4)
        println!("Sum: {}", pi.sum());  // 7 (only available for i32)
    }

Associated Types
----------------

Associated types are type placeholders within traits that implementors must specify.
They're similar to C++ template type aliases but are part of the trait definition.
Associated types are preferred over generic traits when there should be only one
implementation per type:

**C++:**

.. code-block:: cpp

    #include <iostream>
    #include <vector>

    // Template with associated type via typedef
    template<typename Derived>
    class Container {
    public:
      // Associated type defined by derived class
      using Item = typename Derived::ItemType;

      virtual Item* get(size_t index) = 0;
      virtual size_t len() const = 0;
      virtual ~Container() = default;
    };

    class IntVec : public Container<IntVec> {
      std::vector<int> data_;
    public:
      using ItemType = int;

      IntVec(std::initializer_list<int> init) : data_(init) {}

      int* get(size_t index) override {
        return index < data_.size() ? &data_[index] : nullptr;
      }

      size_t len() const override { return data_.size(); }
    };

    int main() {
      IntVec v = {1, 2, 3, 4, 5};
      std::cout << "Length: " << v.len() << "\n";
      if (auto* item = v.get(2)) {
        std::cout << "Item at 2: " << *item << "\n";
      }
      return 0;
    }

**Rust:**

.. code-block:: rust

    // Trait with associated type
    trait Container {
        type Item;  // associated type - implementor specifies this

        fn get(&self, index: usize) -> Option<&Self::Item>;
        fn len(&self) -> usize;
        fn is_empty(&self) -> bool {
            self.len() == 0
        }
    }

    // Wrapper around Vec<i32>
    struct IntVec(Vec<i32>);

    impl Container for IntVec {
        type Item = i32;  // specify the associated type

        fn get(&self, index: usize) -> Option<&i32> {
            self.0.get(index)
        }

        fn len(&self) -> usize {
            self.0.len()
        }
    }

    // Generic wrapper - associated type depends on T
    struct GenericVec<T>(Vec<T>);

    impl<T> Container for GenericVec<T> {
        type Item = T;

        fn get(&self, index: usize) -> Option<&T> {
            self.0.get(index)
        }

        fn len(&self) -> usize {
            self.0.len()
        }
    }

    fn main() {
        let v = IntVec(vec![1, 2, 3, 4, 5]);
        println!("Length: {}", v.len());
        if let Some(item) = v.get(2) {
            println!("Item at 2: {}", item);
        }

        let gv = GenericVec(vec!["a", "b", "c"]);
        println!("Generic length: {}", gv.len());
    }

Common Standard Traits
----------------------

+------------------+----------------------------------+------------------------+
| Rust Trait       | Purpose                          | C++ Equivalent         |
+==================+==================================+========================+
| ``Clone``        | Explicit deep copy               | Copy constructor       |
+------------------+----------------------------------+------------------------+
| ``Copy``         | Implicit bitwise copy            | Trivially copyable     |
+------------------+----------------------------------+------------------------+
| ``Drop``         | Destructor                       | Destructor             |
+------------------+----------------------------------+------------------------+
| ``Default``      | Default value                    | Default constructor    |
+------------------+----------------------------------+------------------------+
| ``Debug``        | Debug formatting                 | operator<< (debug)     |
+------------------+----------------------------------+------------------------+
| ``Display``      | User-facing formatting           | operator<<             |
+------------------+----------------------------------+------------------------+
| ``PartialEq``    | Equality comparison              | operator==             |
+------------------+----------------------------------+------------------------+
| ``Eq``           | Total equality                   | operator== (reflexive) |
+------------------+----------------------------------+------------------------+
| ``PartialOrd``   | Partial ordering                 | operator<              |
+------------------+----------------------------------+------------------------+
| ``Ord``          | Total ordering                   | operator<=> (C++20)    |
+------------------+----------------------------------+------------------------+
| ``Hash``         | Hashing                          | std::hash              |
+------------------+----------------------------------+------------------------+
| ``From/Into``    | Type conversion                  | Conversion constructor |
+------------------+----------------------------------+------------------------+
| ``Iterator``     | Iteration                        | begin()/end()          |
+------------------+----------------------------------+------------------------+

Deriving Traits
~~~~~~~~~~~~~~~

.. code-block:: rust

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
    struct Point {
        x: i32,
        y: i32,
    }

    fn main() {
        let p1 = Point { x: 1, y: 2 };
        let p2 = p1.clone();
        println!("{:?}", p1);        // Debug
        println!("{}", p1 == p2);    // PartialEq
    }

Trait Objects (Dynamic Dispatch)
--------------------------------

**C++ (virtual functions):**

.. code-block:: cpp

    class Shape {
    public:
      virtual double area() const = 0;
      virtual ~Shape() = default;
    };

    void print_area(const Shape& shape) {
      std::cout << shape.area();
    }

**Rust (dyn Trait):**

.. code-block:: rust

    trait Shape {
        fn area(&self) -> f64;
    }

    // Dynamic dispatch with trait object
    fn print_area(shape: &dyn Shape) {
        println!("{}", shape.area());
    }

    // Or with Box for owned trait objects
    fn print_area_boxed(shape: Box<dyn Shape>) {
        println!("{}", shape.area());
    }

Static vs Dynamic Dispatch
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: rust

    trait Draw {
        fn draw(&self);
    }

    // Static dispatch (monomorphization, like C++ templates)
    fn draw_static<T: Draw>(item: &T) {
        item.draw();
    }

    // Dynamic dispatch (vtable, like C++ virtual)
    fn draw_dynamic(item: &dyn Draw) {
        item.draw();
    }

Implementing External Traits
----------------------------

Rust's orphan rule: you can implement a trait for a type only if either the trait
or the type is defined in your crate.

.. code-block:: rust

    use std::fmt::Display;

    struct Wrapper(Vec<String>);

    // Can implement Display for our Wrapper type
    impl Display for Wrapper {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "[{}]", self.0.join(", "))
        }
    }

See Also
--------

- :doc:`rust_error` - Result and Option traits
- :doc:`rust_iterator` - Iterator trait
