=============
Type Erasure
=============

.. meta::
   :description: C++ type erasure guide covering manual Concept/Model wrappers, small-buffer optimization, std::function, std::any, std::move_only_function, performance trade-offs, and the comparison with Rust trait objects.
   :keywords: C++, type erasure, concept, model, AnyDrawable, std::function, std::any, std::move_only_function, polymorphic memory resources, small buffer optimization, SBO, dyn Trait, Rust trait objects, virtual table, runtime polymorphism, value semantics

.. contents:: Table of Contents
    :backlinks: none

Type erasure is the third flavor of polymorphism in modern C++ — alongside
inheritance-based dispatch (see :doc:`cpp_polymorphism`) and template-based
static dispatch (see :doc:`cpp_template`, :doc:`cpp_concepts`). It lets a
*single non-templated value type* hold any concrete object that satisfies a
behavioral contract, without forcing those objects to inherit from a common
base class.

The standard library uses this technique extensively: ``std::function``,
``std::any``, ``std::move_only_function`` (C++23), and the polymorphic
allocators in ``<memory_resource>`` are all type-erased wrappers. The same
technique lets you build your own value-semantic interfaces.

In Rust, the equivalent is ``Box<dyn Trait>``: an opaque pointer to any
implementor of a trait, with a hidden vtable. C++ reaches the same end with a
hand-rolled wrapper because there is no language-level construct for it. See
:doc:`../rust/rust_traits` for the Rust side.

Why Type Erasure
----------------

Inheritance-based polymorphism couples behavior to type identity:

.. code-block:: cpp

    struct Drawable {                          // contract …
      virtual std::string draw() const = 0;
      virtual ~Drawable() = default;
    };
    struct Circle : Drawable { /* ... */ };    // … forces inheritance.
    struct Square : Drawable { /* ... */ };

This is fine for closed sets you control. It breaks down when:

* A third-party library exports ``Circle`` and you cannot change it.
* You want value semantics — copy, move, store-in-vector — without
  ``unique_ptr<Drawable>`` plumbing on every call site.
* Two unrelated codebases each want to plug into the same interface.

Type erasure removes the inheritance requirement: any type with the right
shape (``draw()`` returning ``std::string``) can be wrapped without
modification.

The Concept / Model Pattern
---------------------------

:Source: `src/type-erasure/any-drawable <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/type-erasure/any-drawable>`_

The standard recipe: a *Concept* abstract class declares the operations, a
*Model* template implements them for any concrete type, and the public
wrapper holds a ``unique_ptr<Concept>``.

.. code-block:: cpp

    class AnyDrawable {
      // 1. Internal interface — never seen by callers.
      struct Concept {
        virtual ~Concept() = default;
        virtual std::string do_draw() const = 0;
        virtual std::unique_ptr<Concept> clone() const = 0;
      };
      // 2. Templated bridge from any T to the Concept interface.
      template <typename T>
      struct Model : Concept {
        T value;
        explicit Model(T v) : value(std::move(v)) {}
        std::string do_draw() const override { return value.draw(); }
        std::unique_ptr<Concept> clone() const override {
          return std::make_unique<Model>(*this);
        }
      };
      std::unique_ptr<Concept> p_;
     public:
      // 3. Type-erasing constructor.
      template <typename T>
      AnyDrawable(T x) : p_(std::make_unique<Model<T>>(std::move(x))) {}
      // 4. Value semantics: deep copy via clone().
      AnyDrawable(const AnyDrawable &o) : p_(o.p_->clone()) {}
      AnyDrawable(AnyDrawable &&) noexcept = default;
      // 5. Public API forwards to the hidden vtable.
      std::string draw() const { return p_->do_draw(); }
    };

Now any type with ``.draw() -> std::string`` works:

.. code-block:: cpp

    struct Circle { double r; std::string draw() const { return "circle"; } };
    struct Square { double s; std::string draw() const { return "square"; } };

    std::vector<AnyDrawable> shapes;
    shapes.emplace_back(Circle{1.0});
    shapes.emplace_back(Square{2.0});
    for (const auto &s : shapes) std::cout << s.draw() << "\n";

Notice that ``Circle`` and ``Square`` know nothing about ``AnyDrawable``;
they just have to provide ``draw()``.

Small Buffer Optimization (SBO)
-------------------------------

:Source: `src/type-erasure/sbo <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/type-erasure/sbo>`_

The naive wrapper above always heap-allocates. For small, frequently-copied
types (lambdas, function pointers, integer-shaped state) the allocation
dominates the call cost. The fix is a *small buffer optimization*: reserve a
fixed-size aligned buffer in the wrapper, store small types in-place, fall
back to the heap for everything else.

.. code-block:: cpp

    class AnyDrawable {
      static constexpr std::size_t kBuf = 32;

      struct VTable {
        std::string (*draw)(const void *);
        void (*copy)(void *dst, const void *src);
        void (*move)(void *dst, void *src) noexcept;
        void (*destroy)(void *) noexcept;
        bool inline_storage;
      };
      // One VTable per stored T, picked by `vtable_for<T>()` based on whether
      // T fits in `kBuf` and is nothrow-move-constructible.

      alignas(std::max_align_t) std::byte storage_[kBuf];
      const VTable *vt_ = nullptr;
      // ... constructors / destructor dispatch through `vt_`.
    };

Trade-offs:

* **Pro:** small types dispatch through a single indirection with no
  allocator traffic; cache locality improves dramatically.
* **Con:** ``sizeof(AnyDrawable)`` grows by ``kBuf``; the wrapper is no
  longer a thin pointer.
* **Con:** SBO interacts badly with type-changing assignment if you do not
  destroy the old value in place before constructing the new one.

The standard library's ``std::function`` implementations all use SBO; the
buffer size is implementation-defined (libc++ historically uses a few
pointers' worth, MSVC slightly more). Do not rely on a particular cutoff.

std::function
-------------

:Source: `src/type-erasure/std-function <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/type-erasure/std-function>`_

``std::function<R(Args...)>`` is the canonical type-erased *callable*
wrapper. It accepts function pointers, lambdas (with or without captures),
member function pointers, and any object with a matching ``operator()``.

.. code-block:: cpp

    std::function<int(int, int)> f;
    f = [](int a, int b) { return a + b; };
    f = std::plus<int>{};

Three things to remember about ``std::function``:

* **Copyable target requirement.** The stored callable must be
  ``CopyConstructible``. This rules out lambdas that capture
  ``std::unique_ptr`` by value. Workarounds: capture a ``shared_ptr``, or
  use ``std::move_only_function`` (C++23, see below).
* **It is not free.** A virtual-style indirection per call, plus possibly an
  allocation if the callable does not fit in the SBO. On a hot path this
  matters; ``std::function_ref`` (proposed) or a hand-rolled wrapper does
  not.
* **Type erasure of the call signature, not the callable type.** Two
  ``std::function<int(int)>`` values can hold completely different
  underlying types — that is the whole point — but they are not comparable
  for equality.

std::any
--------

:Source: `src/type-erasure/std-any <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/type-erasure/std-any>`_

``std::any`` (C++17) erases the *type* of a value but not any behavior. You
can store anything copyable, then ``any_cast`` it back to the original type
to do anything useful.

.. code-block:: cpp

    std::any a = 42;
    int x = std::any_cast<int>(a);                  // OK
    std::string s = std::any_cast<std::string>(a);  // throws std::bad_any_cast

    int *p = std::any_cast<int>(&a);                // pointer form: nullptr on mismatch

Use ``std::any`` for opaque payloads (configuration values, attached
properties, plugin interop) where the consumer knows the type at the
retrieval point. If you want to *call* methods polymorphically without
knowing the underlying type, you want a Concept/Model wrapper instead —
``std::any`` cannot do that.

std::move_only_function (C++23)
-------------------------------

:Source: `src/type-erasure/move-only-function <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/type-erasure/move-only-function>`_

``std::move_only_function`` lifts the ``CopyConstructible`` requirement so
move-only callables fit:

.. code-block:: cpp

    std::move_only_function<int()> f =
        [p = std::make_unique<int>(42)] { return *p; };  // OK
    auto g = std::move(f);                                // copy: ill-formed

It is move-constructible and move-assignable but not copyable. Use it for
one-shot tasks, channel sends, and anywhere the captured state is itself
move-only (most heap-owning resources). Where ``std::function`` would force
you to ``shared_ptr`` the captured state, ``std::move_only_function`` lets
you keep the original ownership semantics.

When to Reach for Type Erasure
------------------------------

Choose type erasure when:

* You need polymorphic *value* semantics — copies, containers, pass-by-value
  — without ``unique_ptr<Base>`` plumbing.
* The set of concrete types is open or owned by other people.
* You want the call site to take a single concrete type rather than a
  template parameter.

Choose virtual functions (see :doc:`cpp_polymorphism`) when:

* The hierarchy is closed and you control all derived types.
* Inheritance-based modeling is already idiomatic in the codebase.

Choose CRTP / concepts (see :doc:`cpp_template`, :doc:`cpp_concepts`) when:

* All types are known at compile time.
* You want zero-overhead dispatch and full inlining.
* The interface is small and the ABI surface is your own translation unit.

Performance Notes
-----------------

* Each type-erased call costs **one indirect call** (vtable lookup) — same
  order as a virtual call.
* Without SBO, construction and copy include a heap allocation. SBO removes
  that for small targets.
* The compiler usually cannot inline through the vtable. Hot inner loops
  should not call through ``AnyDrawable`` or ``std::function``; pull the
  call out of the loop or use a static interface.
* Stack frames in profilers may show a generic ``Model<T>::do_draw`` with
  ``T`` mangled; debug symbols help.

Common Pitfalls
---------------

* **Forgetting** ``clone()`` **on the Concept.** Without it the wrapper is
  move-only — fine if intentional, surprising if not.
* **Slicing on construction.** ``AnyDrawable(T x)`` takes ``T`` by value,
  which copies. To avoid the copy, write ``AnyDrawable(T &&x)`` and forward.
* **Lifetime bugs in non-owning erasers.** A ``function_ref``-style wrapper
  that stores a raw pointer to its target dangles if the target dies first.
  Owning wrappers (``std::function``, the example above) avoid this; refs
  are an explicit performance trade.
* **Hidden allocation in tight loops.** A ``std::function`` re-assigned in
  a loop may allocate on each assignment if the new callable does not fit in
  the SBO. Reuse the wrapper, or move it.
* **Mixing copy-only and move-only erasers.** ``std::function`` cannot hold
  a move-only lambda. Either capture by ``shared_ptr`` or switch to
  ``std::move_only_function``.

See Also
--------

* :doc:`cpp_polymorphism` — virtual functions, the inheritance-based flavor
* :doc:`cpp_template` — CRTP and template-based static dispatch
* :doc:`cpp_concepts` — modern constraints replacing SFINAE
* :doc:`../rust/rust_traits` — Rust trait objects (``dyn Trait``), the
  language-level analog
