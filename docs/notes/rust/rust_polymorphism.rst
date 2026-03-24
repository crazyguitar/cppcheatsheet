============
Polymorphism
============

.. meta::
   :description: Rust polymorphism for C++ developers covering trait objects, dynamic dispatch, enum dispatch, and object safety with side-by-side C++ comparisons.
   :keywords: Rust, C++, polymorphism, trait objects, dyn, vtable, dynamic dispatch, static dispatch, enum dispatch, object safety

.. contents:: Table of Contents
    :backlinks: none

:Source: `src/rust/polymorphism <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/polymorphism>`_

Rust achieves polymorphism without inheritance. Where C++ uses virtual functions and
class hierarchies, Rust provides two main approaches: trait objects (dynamic dispatch)
and enums (closed-set dispatch). Both avoid the fragile base class problem.

Trait Objects (Dynamic Dispatch)
--------------------------------

Trait objects (``&dyn Trait`` or ``Box<dyn Trait>``) are Rust's equivalent of C++
virtual function calls. They use a vtable for runtime dispatch.

**C++ (virtual):**

.. code-block:: cpp

    #include <iostream>
    #include <memory>
    #include <vector>

    class Shape {
    public:
      virtual double area() const = 0;
      virtual const char* name() const = 0;
      virtual ~Shape() = default;
    };

    class Circle : public Shape {
      double radius_;
    public:
      Circle(double r) : radius_(r) {}
      double area() const override { return 3.14159265 * radius_ * radius_; }
      const char* name() const override { return "Circle"; }
    };

    class Rectangle : public Shape {
      double w_, h_;
    public:
      Rectangle(double w, double h) : w_(w), h_(h) {}
      double area() const override { return w_ * h_; }
      const char* name() const override { return "Rectangle"; }
    };

    void print_area(const Shape& s) {
      std::cout << s.name() << ": " << s.area() << "\n";
    }

    int main() {
      std::vector<std::unique_ptr<Shape>> shapes;
      shapes.push_back(std::make_unique<Circle>(3.0));
      shapes.push_back(std::make_unique<Rectangle>(4.0, 5.0));
      for (auto& s : shapes) print_area(*s);
    }

**Rust:**

.. code-block:: rust

    trait Shape {
        fn area(&self) -> f64;
        fn name(&self) -> &str;
    }

    struct Circle { radius: f64 }
    struct Rectangle { width: f64, height: f64 }

    impl Shape for Circle {
        fn area(&self) -> f64 { std::f64::consts::PI * self.radius * self.radius }
        fn name(&self) -> &str { "Circle" }
    }

    impl Shape for Rectangle {
        fn area(&self) -> f64 { self.width * self.height }
        fn name(&self) -> &str { "Rectangle" }
    }

    fn print_area(shape: &dyn Shape) {
        println!("{}: area = {:.2}", shape.name(), shape.area());
    }

    fn main() {
        let shapes: Vec<Box<dyn Shape>> = vec![
            Box::new(Circle { radius: 3.0 }),
            Box::new(Rectangle { width: 4.0, height: 5.0 }),
        ];
        for s in &shapes {
            print_area(s.as_ref());
        }
    }

Key differences from C++:

- No inheritance hierarchy — any type can implement any trait
- No virtual destructor needed — ``Box<dyn Trait>`` handles cleanup via ``Drop``
- ``dyn`` keyword makes dynamic dispatch explicit (C++ hides it behind ``virtual``)

Static vs Dynamic Dispatch
--------------------------

Rust lets you choose between static dispatch (monomorphization, like C++ templates)
and dynamic dispatch (vtable, like C++ virtual) per call site.

.. code-block:: rust

    // Static dispatch — compiler generates specialized code per type
    // Equivalent to C++ templates: zero overhead, but larger binary
    fn print_area_static<T: Shape>(shape: &T) {
        println!("{}: area = {:.2}", shape.name(), shape.area());
    }

    // Dynamic dispatch — single function, vtable lookup at runtime
    // Equivalent to C++ virtual: smaller binary, slight runtime cost
    fn print_area_dynamic(shape: &dyn Shape) {
        println!("{}: area = {:.2}", shape.name(), shape.area());
    }

.. list-table::
   :header-rows: 1

   * - Aspect
     - Static (``impl Trait`` / generics)
     - Dynamic (``dyn Trait``)
   * - C++ equivalent
     - Templates
     - Virtual functions
   * - Dispatch
     - Compile-time
     - Runtime (vtable)
   * - Performance
     - Zero-cost, inlinable
     - Indirect call overhead
   * - Binary size
     - Larger (monomorphized copies)
     - Smaller (single function)
   * - Heterogeneous collections
     - No
     - Yes

Enum-based Dispatch
-------------------

When the set of variants is known at compile time, enums provide a closed-set
alternative to trait objects. This avoids heap allocation and vtable overhead.

**C++ (variant):**

.. code-block:: cpp

    #include <iostream>
    #include <variant>
    #include <string>

    struct Dog { std::string name; };
    struct Cat { std::string name; };

    using Animal = std::variant<Dog, Cat>;

    const char* speak(const Animal& a) {
      return std::visit([](auto& v) -> const char* {
        if constexpr (std::is_same_v<std::decay_t<decltype(v)>, Dog>) return "Woof!";
        else return "Meow!";
      }, a);
    }

**Rust:**

.. code-block:: rust

    enum Animal {
        Dog(String),
        Cat(String),
    }

    impl Animal {
        fn speak(&self) -> &str {
            match self {
                Animal::Dog(_) => "Woof!",
                Animal::Cat(_) => "Meow!",
            }
        }
    }

Advantages over trait objects:

- Stack-allocated, no ``Box`` needed
- Exhaustive ``match`` — compiler warns if you miss a variant
- Better cache locality

Returning Trait Objects
-----------------------

Functions can return different concrete types via ``Box<dyn Trait>``, similar to
returning ``std::unique_ptr<Base>`` in C++.

.. code-block:: rust

    fn make_shape(kind: &str) -> Box<dyn Shape> {
        match kind {
            "circle" => Box::new(Circle { radius: 5.0 }),
            _ => Box::new(Rectangle { width: 4.0, height: 3.0 }),
        }
    }

Trait Object References (``&dyn Trait``)
----------------------------------------

A ``&dyn Trait`` is a **fat pointer** — two machine words (16 bytes on 64-bit):
one pointer to the data, one pointer to the vtable. No heap allocation is involved;
it simply borrows an existing value.

This is the lightest way to do dynamic dispatch:

.. code-block:: rust

    fn total_area(shapes: &[&dyn Shape]) -> f64 {
        shapes.iter().map(|s| s.area()).sum()
    }

    let c = Circle { radius: 1.0 };
    let r = Rectangle { width: 2.0, height: 3.0 };
    let refs: Vec<&dyn Shape> = vec![&c, &r];  // no Box, no heap
    println!("{}", total_area(&refs));

In C++, the equivalent is passing ``const Shape&`` — but C++ references are thin
pointers (the vtable pointer lives inside the object). Rust's fat pointer keeps the
vtable external, which is why ``dyn`` is needed to opt in.

``Rc<RefCell<dyn Trait>>`` — Shared Mutable Trait Objects
---------------------------------------------------------

When you need **shared ownership** and **interior mutability** with trait objects,
combine ``Rc`` (reference counting) with ``RefCell`` (runtime borrow checking):

.. code-block:: rust

    use std::cell::RefCell;
    use std::rc::Rc;

    trait Counter {
        fn increment(&mut self);
        fn count(&self) -> u32;
    }

    struct ClickCounter { clicks: u32 }

    impl Counter for ClickCounter {
        fn increment(&mut self) { self.clicks += 1; }
        fn count(&self) -> u32 { self.clicks }
    }

    let counters: Vec<Rc<RefCell<dyn Counter>>> = vec![
        Rc::new(RefCell::new(ClickCounter { clicks: 0 })),
    ];

    let shared = Rc::clone(&counters[0]);  // second owner
    shared.borrow_mut().increment();       // mutate through RefCell
    counters[0].borrow_mut().increment();
    assert_eq!(counters[0].borrow().count(), 2);

**C++ equivalent:** ``std::shared_ptr<Shape>`` — but C++ doesn't distinguish shared
ownership from mutability. Rust forces you to be explicit:

.. list-table::
   :header-rows: 1

   * - Rust
     - C++ equivalent
     - Use case
   * - ``Box<dyn Trait>``
     - ``unique_ptr<Base>``
     - Single owner, heap-allocated
   * - ``&dyn Trait``
     - ``const Base&``
     - Borrowed reference, no allocation
   * - ``Rc<dyn Trait>``
     - ``shared_ptr<const Base>``
     - Shared ownership, immutable
   * - ``Rc<RefCell<dyn Trait>>``
     - ``shared_ptr<Base>``
     - Shared ownership, mutable at runtime

.. note::

   For multithreaded code, replace ``Rc`` with ``Arc`` and ``RefCell`` with
   ``Mutex`` or ``RwLock``: ``Arc<Mutex<dyn Trait + Send>>``.

``Rc<RefCell<Box<dyn Trait>>>`` — Shared Mutable Trait Objects with Box
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You may see a ``Box`` inside the ``RefCell``. This pattern appears in real
codebases (e.g. `this example <https://github.com/crazyguitar/nes/blob/812f595/src/core/bus.rs#L31>`_):

.. code-block:: rust

    type SharedLogger = Rc<RefCell<Box<dyn Logger>>>;

Why the extra ``Box``? Each layer serves a distinct purpose:

.. code-block:: text

    Rc<            -- shared ownership (multiple owners hold a clone)
      RefCell<     -- interior mutability (borrow checked at runtime)
        Box<       -- heap-allocate the trait object (dyn Trait is unsized)
          dyn Logger
        >
      >
    >

On modern Rust (2021+), ``Rc<RefCell<dyn Trait>>`` works directly because ``Rc``
can hold unsized types via ``CoerceUnsized``. The ``Box`` layer is still common
because:

- Factory functions naturally return ``Box<dyn Trait>`` which slots right in
- Older codebases (pre-2021 edition) required it

.. code-block:: rust

    use std::cell::RefCell;
    use std::rc::Rc;

    trait Logger {
        fn log(&mut self, msg: &str);
        fn entries(&self) -> &[String];
    }

    struct ConsoleLogger { logs: Vec<String> }

    impl Logger for ConsoleLogger {
        fn log(&mut self, msg: &str) { self.logs.push(msg.to_string()); }
        fn entries(&self) -> &[String] { &self.logs }
    }

    type SharedLogger = Rc<RefCell<Box<dyn Logger>>>;

    fn create_logger() -> SharedLogger {
        let logger: Box<dyn Logger> = Box::new(ConsoleLogger { logs: vec![] });
        Rc::new(RefCell::new(logger))
    }

    let logger = create_logger();
    let writer = Rc::clone(&logger);   // same type, refcount = 2
    let reader = Rc::clone(&logger);   // same type, refcount = 3

    // All three are Rc<RefCell<Box<dyn Logger>>>.
    // Rc::clone does NOT unwrap a layer — it bumps the reference count.
    // RefCell::borrow_mut() provides &mut access checked at runtime.
    writer.borrow_mut().log("hello");
    reader.borrow_mut().log("world");
    assert_eq!(logger.borrow().entries().len(), 2);

How ``borrow_mut()`` provides mutability — the ``Box<dyn Logger>`` itself is not
mutable. ``RefCell`` is what enables mutation through a shared reference:

.. code-block:: text

    writer                                // Rc<RefCell<Box<dyn Logger>>>
        .borrow_mut()                     // RefCell -> RefMut<Box<dyn Logger>>
        .log("hello")                     // auto-deref: &mut Box<dyn Logger> -> &mut dyn Logger

``Rc::clone`` does **not** unwrap a layer. All clones have the same type
(``Rc<RefCell<Box<dyn Logger>>>``). It only increments the reference count.

Object Safety
-------------

Not all traits can be used as trait objects. A trait is object-safe if:

- No methods return ``Self``
- No methods have generic type parameters
- All methods have a receiver (``&self``, ``&mut self``, ``self``, etc.)

.. code-block:: rust

    // Object-safe — can use as `dyn Drawable`
    trait Drawable {
        fn draw(&self);
    }

    // NOT object-safe — returns Self
    trait Clonable {
        fn clone(&self) -> Self;
    }

    // NOT object-safe — generic method
    trait Converter {
        fn convert<T>(&self) -> T;
    }

The ``Clone`` trait in std is not object-safe, which is why you cannot write
``Box<dyn Clone>``. Use workarounds like a helper trait when needed.

See Also
--------

- :doc:`rust_traits` - Trait definitions, bounds, and deriving
- :doc:`rust_smartptr` - Box, Rc, Arc for trait object storage
- :doc:`rust_casting` - Type conversions and ``Any`` trait
