==========================
Polymorphism & Inheritance
==========================

.. meta::
   :description: C++ polymorphism guide covering virtual functions, vtable layout, override and final, virtual destructors, abstract classes, multiple and virtual inheritance, the slicing problem, static vs dynamic dispatch, and the runtime cost of virtual calls.
   :keywords: C++, polymorphism, virtual function, vtable, vptr, override, final, virtual destructor, abstract class, pure virtual, multiple inheritance, virtual inheritance, diamond problem, slicing, static dispatch, dynamic dispatch, devirtualization, RTTI, dynamic_cast, CRTP

.. contents:: Table of Contents
    :backlinks: none

Polymorphism lets a single interface drive different concrete behaviors.
Modern C++ supports three distinct flavors:

* **Subtype polymorphism** — inheritance and ``virtual`` functions, resolved
  at run time through a per-object pointer to a table of function pointers
  (the vtable). Concrete types share a common base class.
* **Static (parametric) polymorphism** — templates, overloading, CRTP, and
  concepts, resolved at compile time. Zero indirection, full inlining.
* **Trait-style polymorphism** — type erasure. Unrelated concrete types are
  stored behind a value-semantic wrapper that calls into a hidden vtable.
  No shared base class is required; the contract is purely behavioral. This
  is the C++ analog of Rust's ``dyn Trait``.

This chapter focuses on the first flavor, the vtable mechanics that implement
it, and the pitfalls of class hierarchies — then revisits CRTP and type
erasure at the end so you can pick the right tool.

For the static counterpart — CRTP, concepts, and template-based dispatch — see
:doc:`cpp_template` and :doc:`cpp_concepts`. For Rust's trait model, see
:doc:`../rust/rust_traits`. For the safe runtime-checked downcast, see
:doc:`cpp_casting`.

Virtual Functions
-----------------

:Source: `src/polymorphism/virtual-functions <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/polymorphism/virtual-functions>`_

A member function declared ``virtual`` in a base class can be overridden in a
derived class. When called through a pointer or reference to the base, the
*dynamic* type of the object decides which override runs. Without ``virtual``,
the call is bound to the *static* type at compile time.

.. code-block:: cpp

    #include <iostream>

    struct Animal {
      virtual void speak() const { std::cout << "..." << "\n"; }
      void name() const { std::cout << "Animal" << "\n"; }  // non-virtual
    };

    struct Dog : Animal {
      void speak() const override { std::cout << "Woof" << "\n"; }
      void name() const { std::cout << "Dog" << "\n"; }
    };

    int main() {
      Dog d;
      Animal &a = d;
      a.speak();   // "Woof"   - dynamic dispatch through vtable
      a.name();    // "Animal" - static dispatch, base version called
    }

The non-virtual ``name`` is *hidden* in ``Dog``, not overridden. The choice of
which to call is made by the compiler from the static type, so ``a.name()``
calls ``Animal::name`` even though ``a`` actually refers to a ``Dog``.

override and final
------------------

:Source: `src/polymorphism/override-final <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/polymorphism/override-final>`_

``override`` (C++11) tells the compiler that a function is intended to override
a base virtual. If the signature does not actually match a base virtual, the
program fails to compile. This catches typos, forgotten ``const`` qualifiers,
and signature drift after a refactor.

``final`` prevents further overriding (when applied to a function) or further
derivation (when applied to a class). It also enables a useful optimization:
the compiler can devirtualize calls to a ``final`` override.

.. code-block:: cpp

    struct Base {
      virtual void f() const;
      virtual void g(int);
    };

    struct Derived : Base {
      void f() const override;          // OK
      // void f() override;             // ERROR: missing const
      void g(int) override final;       // cannot be overridden further
    };

    struct Sealed final : Derived {     // cannot be derived from
      void f() const override;
    };

Always write ``override`` on overriding functions. It is not redundant — it is
a compile-time contract that the override still tracks the base.

Pure Virtual Functions and Abstract Classes
-------------------------------------------

:Source: `src/polymorphism/pure-virtual <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/polymorphism/pure-virtual>`_

A function with ``= 0`` is *pure virtual*; the class that declares it is
*abstract* and cannot be instantiated. Derived classes must override every
pure virtual function before they become concrete.

.. code-block:: cpp

    struct Shape {
      virtual double area() const = 0;
      virtual ~Shape() = default;       // see Virtual Destructors
    };

    struct Circle : Shape {
      explicit Circle(double r) : r_(r) {}
      double area() const override { return 3.14159 * r_ * r_; }
     private:
      double r_;
    };

    // Shape s;        // ERROR: cannot instantiate abstract class
    Circle c(1.0);     // OK
    Shape *p = &c;     // OK: pointer to abstract base

A pure virtual function can still have a definition; derived classes that
explicitly call ``Base::fn()`` will then run that definition. This is
occasionally useful for shared default behavior in an otherwise abstract
interface.

Virtual Destructors
-------------------

:Source: `src/polymorphism/virtual-destructor <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/polymorphism/virtual-destructor>`_

If a class is meant to be a polymorphic base — that is, deleted through a
base pointer — its destructor **must** be ``virtual``. Without it, deleting
through the base pointer invokes only the base destructor, leaking the
derived part of the object. This is undefined behavior.

.. code-block:: cpp

    struct Base {
      ~Base() { /* not virtual */ }
    };

    struct Derived : Base {
      std::vector<int> data_{1, 2, 3};
      ~Derived() { /* destructor never runs */ }
    };

    Base *p = new Derived;
    delete p;   // UB: ~Derived not called, data_ leaks

Two safe defaults:

* If a class has any virtual function, give it a ``virtual ~Base() = default;``.
* If a class is **not** intended for polymorphic deletion, mark it ``final``
  or document that fact, so callers do not silently rely on a non-virtual
  destructor.

The vtable and vptr
-------------------

Most compilers implement dynamic dispatch with a *vtable*: a per-class array
of function pointers, one slot per virtual function. Each polymorphic object
stores a hidden *vptr* pointing to its class's vtable. A virtual call becomes
two indirections: load the vptr, index into the vtable, jump to the target.

.. code-block:: cpp

    struct B {
      virtual void f();          // vtable slot 0
      virtual void g();          // vtable slot 1
      int x;
    };

    struct D : B {
      void f() override;         // overrides slot 0
      void h();                  // not virtual
      int y;
    };

    // Memory layout (typical Itanium ABI):
    //
    //   B object:        D object:
    //   +-------+        +-------+
    //   | vptr  |  ----> | vptr  |  ----> D's vtable: [&D::f, &B::g, ...]
    //   |   x   |        |   x   |
    //   +-------+        |   y   |
    //                    +-------+

Consequences of this layout:

* Every polymorphic object pays one pointer (typically 8 bytes on 64-bit) of
  storage for the vptr.
* A virtual call is one extra load and an indirect branch compared to a
  direct call. The cost is small but real, and it can defeat inlining.
* The vptr is set by each constructor as the object's type changes during
  construction (see Construction and Destruction Order).

The exact layout is ABI-defined. The Itanium ABI used by GCC, Clang, and the
Apple toolchain places the vptr at offset 0 and shares vtables between
translation units; MSVC's ABI differs in the details (and in how it handles
multiple inheritance) but follows the same general scheme.

You can inspect the layout with ``-fdump-lang-class`` (GCC) or
``-Xclang -fdump-record-layouts`` (Clang).

Static vs. Dynamic Dispatch
---------------------------

A non-virtual call is bound at compile time and can be inlined. A virtual call
requires the dynamic type, so it can only be devirtualized when the compiler
*proves* the dynamic type — for example, when the object's most-derived type
is visible at the call site, or when the function is declared ``final``.

.. code-block:: cpp

    struct I { virtual int f() const = 0; };
    struct A final : I { int f() const override { return 1; } };

    int callI(const I &i)  { return i.f(); }   // virtual call
    int callA(const A &a)  { return a.f(); }   // devirtualized: A is final
    int callD()            { A a; return a.f(); } // direct call, inlinable

Rules of thumb:

* Calls through a base pointer or reference are virtual.
* Calls on a known concrete object are direct.
* ``final`` on the override (or the class) lets the compiler devirtualize.
* Inside a constructor or destructor, virtual calls dispatch to the
  *currently-constructed* type, not the most-derived type (see below).

Construction and Destruction Order
----------------------------------

:Source: `src/polymorphism/ctor-dtor-dispatch <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/polymorphism/ctor-dtor-dispatch>`_

During construction, the object's vptr is set to each base subobject's vtable
in turn before its own constructor body runs. So a virtual call from inside a
base constructor dispatches to the base version, not the derived override —
the derived part of the object does not exist yet. Destruction is the mirror
image.

.. code-block:: cpp

    struct B {
      B() { f(); }                              // calls B::f, not D::f
      virtual void f() { std::puts("B::f"); }
      virtual ~B() { f(); }                     // calls B::f
    };

    struct D : B {
      void f() override { std::puts("D::f"); }
    };

    D d;   // prints "B::f" twice (ctor and dtor)

This is a common interview question and a real source of bugs. Never call
virtual functions from a constructor or destructor expecting derived
behavior. If you need polymorphic initialization, use a two-step
``create`` factory or a separate ``init`` call after construction.

The Slicing Problem
-------------------

:Source: `src/polymorphism/slicing <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/polymorphism/slicing>`_

Assigning a derived object to a base *value* copies only the base subobject —
the derived part is "sliced off". The result is a base object that has lost
its dynamic type, and any virtual call on it will run the base version.

.. code-block:: cpp

    struct Animal { virtual void speak() const { std::puts("..."); } };
    struct Dog : Animal { void speak() const override { std::puts("Woof"); } };

    void by_value(Animal a)       { a.speak(); }   // sliced -> "..."
    void by_reference(const Animal &a) { a.speak(); }  // polymorphic -> "Woof"

    int main() {
      Dog d;
      by_value(d);
      by_reference(d);
    }

To preserve polymorphism, pass by reference (``const Animal &``), pointer
(``const Animal *``), or a smart pointer (``std::unique_ptr<Animal>``).
Containers of polymorphic objects must hold pointers, not values:
``std::vector<Animal>`` slices on every ``push_back``;
``std::vector<std::unique_ptr<Animal>>`` does not.

Multiple Inheritance
--------------------

:Source: `src/polymorphism/multiple-inheritance <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/polymorphism/multiple-inheritance>`_

A class may have more than one direct base. Each non-virtual base contributes
its own subobject and (if polymorphic) its own vptr. Casting between base
pointers may adjust the pointer value to land on the right subobject.

.. code-block:: cpp

    struct Drawable { virtual void draw() const = 0; virtual ~Drawable() = default; };
    struct Serializable { virtual void save() const = 0; virtual ~Serializable() = default; };

    struct Widget : Drawable, Serializable {
      void draw() const override;
      void save() const override;
    };

    Widget w;
    Drawable     *d = &w;   // points at Drawable subobject
    Serializable *s = &w;   // points at Serializable subobject (offset!)
    // static_cast<void*>(d) != static_cast<void*>(s) in general

Multiple inheritance is fine for *interface inheritance* (abstract bases with
no state). It gets complicated quickly when two bases share a common
ancestor — the diamond problem.

The Diamond Problem and Virtual Inheritance
-------------------------------------------

:Source: `src/polymorphism/diamond <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/polymorphism/diamond>`_

When a class indirectly inherits from the same base through two paths, the
default is to embed two copies of that base. Member access becomes ambiguous,
and the two copies can diverge.

.. code-block:: cpp

    struct A { int x; };
    struct B : A {};
    struct C : A {};
    struct D : B, C {};   // two A subobjects

    D d;
    // d.x;            // ERROR: ambiguous
    d.B::x = 1;
    d.C::x = 2;        // independent of d.B::x

``virtual`` inheritance collapses the duplicates into a single shared
subobject:

.. code-block:: cpp

    struct A { int x; };
    struct B : virtual A {};
    struct C : virtual A {};
    struct D : B, C {};   // one A subobject

    D d;
    d.x = 1;              // unambiguous

Virtual inheritance has costs: an extra indirection (the *vbase pointer*) to
locate the shared subobject, and the most-derived class becomes responsible
for initializing the virtual base. Reach for it only when you genuinely need
shared state across diamond paths; prefer composition or interface-only
multiple inheritance otherwise.

RTTI and dynamic_cast
---------------------

Run-Time Type Information (RTTI) lets you query the dynamic type of a
polymorphic object. ``typeid`` returns a ``std::type_info``; ``dynamic_cast``
performs a checked downcast.

.. code-block:: cpp

    #include <typeinfo>

    struct Base { virtual ~Base() = default; };
    struct Derived : Base { void special(); };

    void handle(Base *b) {
      if (auto *d = dynamic_cast<Derived *>(b)) {  // null on failure
        d->special();
      }
      try {
        Derived &dr = dynamic_cast<Derived &>(*b); // throws std::bad_cast
        dr.special();
      } catch (const std::bad_cast &) {
      }
    }

``dynamic_cast`` requires a polymorphic source type (at least one virtual
function in the static type). It is not free: it walks the type hierarchy at
run time. If you find yourself reaching for it often, the design probably
wants another virtual function instead.

For the cast hierarchy (``static_cast``, ``const_cast``, ``reinterpret_cast``)
see :doc:`cpp_casting`.

Pure Interfaces and the Non-Virtual Interface Idiom
----------------------------------------------------

:Source: `src/polymorphism/nvi <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/polymorphism/nvi>`_

A common pattern is to expose a non-virtual public API that calls into
private virtual hooks. The base class fixes pre/post conditions; derived
classes only customize behavior.

.. code-block:: cpp

    struct Logger {
      void log(std::string_view msg) {            // public, non-virtual
        prefix();
        write(msg);                                // protected/private virtual
        flush();
      }
      virtual ~Logger() = default;
     private:
      void prefix() { /* ... */ }
      virtual void write(std::string_view) = 0;
      void flush() { /* ... */ }
    };

This *Non-Virtual Interface* (NVI) idiom keeps the contract in one place and
makes the override surface explicit.

Virtual vs. CRTP — When to Pick Which
-------------------------------------

:Source: `src/template/crtp <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/template/crtp>`_

If you do not need heterogeneous containers or runtime dispatch, the
*Curiously Recurring Template Pattern* gives polymorphism without a vtable
and inlines cleanly:

.. code-block:: cpp

    template <typename Derived>
    struct Shape {
      double area() const {
        return static_cast<const Derived &>(*this).area_impl();
      }
    };

    struct Circle : Shape<Circle> {
      double area_impl() const { return 3.14159 * r_ * r_; }
      double r_;
    };

Choose dynamic polymorphism when:

* You need to store mixed concrete types in one collection.
* The set of derived types is not known at the call site.
* The runtime cost of an indirect call is negligible compared to the work
  inside the function.

Choose static polymorphism (CRTP, concepts, templates) when:

* All types are known at compile time.
* The function is small and benefits from inlining.
* You want zero-overhead abstractions.

See :doc:`cpp_template` for CRTP details and :doc:`cpp_concepts` for the
modern concepts-based alternative.

Trait-Style Polymorphism (Type Erasure)
---------------------------------------

The third flavor of polymorphism — the one Rust makes idiomatic with
``dyn Trait`` — is *type erasure*. Unrelated concrete types are stored
behind a value-semantic wrapper that calls into a hidden vtable; no shared
base class is required.

.. code-block:: cpp

    std::vector<AnyDrawable> shapes;
    shapes.emplace_back(Circle{1.0});   // no inheritance from anything
    shapes.emplace_back(Square{2.0});

This pattern is large enough that it has its own chapter:
**see** :doc:`cpp_type_erasure` for the Concept/Model recipe, small-buffer
optimization, ``std::function`` / ``std::any`` /
``std::move_only_function``, and the comparison with Rust trait objects.

Common Pitfalls
---------------

* **Non-virtual destructor in a polymorphic base.** Deleting through a base
  pointer is undefined behavior. Always make the destructor ``virtual`` or
  ``protected`` (the latter prevents base-pointer deletion entirely).
* **Calling virtual functions from constructors or destructors.** Dispatch
  uses the type currently under construction, not the most-derived type.
* **Slicing on copy.** Storing or passing polymorphic objects by value
  silently drops their derived state.
* **Forgetting** ``override``. Without it, a signature change in the base
  silently turns an override into a new function.
* **Default argument values.** Defaults are picked from the *static* type of
  the call, not the dynamic type. ``base->f()`` uses ``Base``'s defaults
  even if the override declares different ones — confusing and a frequent
  source of bugs. Avoid changing defaults in overrides.
* **Calling** ``dynamic_cast`` **in a hot loop.** It walks the type
  hierarchy. Cache the result, or refactor to add a virtual function.
* **Virtual inheritance "just to be safe".** It adds storage and indirection.
  Use it only for genuine diamonds with shared state.
