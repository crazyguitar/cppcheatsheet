===================
Resource Management
===================

.. meta::
   :description: C++ RAII and move semantics guide covering Rule of Zero/Three/Five, value categories, perfect forwarding, noexcept, and common pitfalls with examples.
   :keywords: C++, RAII, move semantics, std::move, std::forward, Rule of Five, Rule of Three, value categories, rvalue, lvalue, perfect forwarding, noexcept, move constructor, move assignment, resource management

.. contents:: Table of Contents
    :backlinks: none

Resource Acquisition Is Initialization (RAII) is a fundamental C++ programming idiom
that ties resource management to object lifetime. When an object is constructed, it
acquires resources; when it is destroyed, it releases them. This pattern eliminates
resource leaks and ensures exception safety by leveraging C++'s deterministic
destruction guarantees.

Note that `Rust <https://www.rust-lang.org/>`_ enforces ownership at compile time, offering strong safety guarantees.
Interestingly, Rust's ownership model closely resembles C++'s RAII and move semantics
(available since C++11). For example, Rust's ownership transfer is analogous to
``std::move``, and its ``Drop`` trait mirrors C++ destructors. This section explores
these resource management techniques in depth.


Special Member Functions
------------------------

:Source: `src/raii/constructors <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/raii/constructors>`_

C++ provides six special member functions that control object lifecycle: default
constructor, destructor, copy constructor, copy assignment operator, move constructor,
and move assignment operator. Understanding when each is called is essential for
implementing correct resource management.

The following example demonstrates all special member functions and the scenarios
that trigger each one. Note how copy operations create independent copies while
move operations transfer ownership, leaving the source in a valid but unspecified state:

.. code-block:: cpp

    #include <iostream>
    #include <utility>

    class Resource {
     public:
      // Constructor: acquires resource
      Resource(int x) : x_(x) {}

      // Default constructor
      Resource() = default;

      // Copy constructor: creates independent copy
      Resource(const Resource &other) : Resource(other.x_) {
        std::cout << "copy constructor\n";
      }

      // Copy assignment: replaces with copy
      Resource &operator=(const Resource &other) {
        std::cout << "copy assignment\n";
        x_ = other.x_;
        return *this;
      }

      // Move constructor: transfers ownership
      Resource(Resource &&other) noexcept : x_(std::move(other.x_)) {
        std::cout << "move constructor\n";
        other.x_ = 0;
      }

      // Move assignment: transfers ownership
      Resource &operator=(Resource &&other) noexcept {
        std::cout << "move assignment\n";
        x_ = std::move(other.x_);
        return *this;
      }

     private:
      int x_ = 0;
    };

    int main(int argc, char *argv[]) {
      Resource r1;                       // default constructor
      Resource r2(1);                    // constructor
      Resource r3 = Resource(2);         // constructor (copy elision)
      Resource r4(r2);                   // copy constructor
      Resource r5(std::move(Resource(2))); // move constructor
      Resource r6 = r1;                  // copy constructor
      Resource r7 = std::move(r3);       // move constructor

      Resource r8, r9;
      r8 = r2;                           // copy assignment
      r9 = std::move(r4);                // move assignment
      r9 = Resource(2);                  // move assignment
    }

Rule of Zero
------------

:Source: `src/raii/rule-of-zero <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/raii/rule-of-zero>`_

The **Rule of Zero** states that classes should not define custom destructors, copy/move
constructors, or copy/move assignment operators. Instead, they should rely on member
objects that manage their own resources (like ``std::string``, ``std::vector``, or
smart pointers). This approach minimizes boilerplate code and reduces the risk of
resource management bugs.

When all members handle their own cleanup, the compiler-generated special member
functions work correctly by default:

.. code-block:: cpp

    #include <iostream>
    #include <string>

    class Document {
     public:
      Document(const std::string &content) : content_(content) {}

      // No destructor, copy/move constructors, or assignment operators needed.
      // The compiler-generated versions correctly handle std::string member.

      friend std::ostream &operator<<(std::ostream &os, const Document &doc);

     private:
      std::string content_;
    };

    std::ostream &operator<<(std::ostream &os, const Document &doc) {
      return os << doc.content_;
    }

    int main(int argc, char *argv[]) {
      Document doc("Rule of Zero");
      std::cout << doc << "\n";
    }

Rule of Three
-------------

:Source: `src/raii/rule-of-three <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/raii/rule-of-three>`_

The **Rule of Three** (C++98) states that if a class requires a user-defined destructor,
copy constructor, or copy assignment operator, it almost certainly requires all three.
This rule applies when a class manages resources that are not automatically handled
by its members (e.g., raw pointers, file handles, network connections).

**What happens without move semantics?** When you define any of the Big Three, the
compiler will NOT generate move constructor or move assignment operator. Instead,
move operations fall back to copy operations. This is safe but inefficient—moving
an object performs a deep copy instead of transferring ownership. For classes
managing expensive resources (large buffers, file handles), this overhead matters.
See :ref:`Rule of Five <rule-of-five>` for the efficient C++11 solution.

Failing to implement all three leads to resource leaks or double-free bugs. In this
example, the class manages a dynamically allocated character array:

.. code-block:: cpp

    #include <cstring>
    #include <iostream>
    #include <memory>
    #include <string>

    class Buffer {
     public:
      Buffer(const char *data, size_t size) : data_(new char[size]), size_(size) {
        std::memcpy(data_, data, size);
      }

      // 1. User-defined destructor: releases resource
      ~Buffer() { delete[] data_; }

      // 2. User-defined copy constructor: deep copy
      Buffer(const Buffer &other) : Buffer(other.data_, other.size_) {}

      // 3. User-defined copy assignment: exception-safe deep copy
      Buffer &operator=(const Buffer &other) {
        if (this == std::addressof(other)) {
          return *this;
        }
        char *new_data = new char[other.size_];  // Allocate first
        std::memcpy(new_data, other.data_, other.size_);
        delete[] data_;  // Then release old resource
        data_ = new_data;
        size_ = other.size_;
        return *this;
      }

      friend std::ostream &operator<<(std::ostream &os, const Buffer &buf);

     private:
      char *data_;
      size_t size_;
    };

    std::ostream &operator<<(std::ostream &os, const Buffer &buf) {
      return os << buf.data_;
    }

    int main(int argc, char *argv[]) {
      std::string s = "Rule of Three";
      Buffer buf(s.c_str(), s.size() + 1);
      std::cout << buf << "\n";
    }

.. _rule-of-five:

Rule of Five
------------

:Source: `src/raii/rule-of-five <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/raii/rule-of-five>`_

The **Rule of Five** (C++11) extends the Rule of Three to include move semantics.
If a class defines any of the five special member functions (destructor, copy
constructor, copy assignment, move constructor, move assignment), it should
explicitly define or delete all five.

Move operations enable efficient transfer of resources without copying—ownership
is transferred in O(1) instead of O(n) deep copy. The ``std::exchange`` utility
is particularly useful for implementing move constructors, as it atomically
replaces a value and returns the old one:

.. code-block:: cpp

    #include <cstring>
    #include <iostream>
    #include <memory>
    #include <string>
    #include <utility>

    class Buffer {
     public:
      Buffer(const char *data, size_t size) : data_(new char[size]), size_(size) {
        std::memcpy(data_, data, size);
      }

      // 1. Destructor
      ~Buffer() { delete[] data_; }

      // 2. Copy constructor
      Buffer(const Buffer &other) : Buffer(other.data_, other.size_) {}

      // 3. Move constructor: transfers ownership using std::exchange
      Buffer(Buffer &&other) noexcept
          : data_(std::exchange(other.data_, nullptr)),
            size_(std::exchange(other.size_, 0)) {}

      // 4. Copy assignment: copy-and-swap idiom
      Buffer &operator=(const Buffer &other) {
        return *this = Buffer(other);
      }

      // 5. Move assignment: swap resources
      Buffer &operator=(Buffer &&other) noexcept {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
        return *this;
      }

      friend std::ostream &operator<<(std::ostream &os, const Buffer &buf);

     private:
      char *data_ = nullptr;
      size_t size_ = 0;
    };

    std::ostream &operator<<(std::ostream &os, const Buffer &buf) {
      return os << buf.data_;
    }

    int main(int argc, char *argv[]) {
      std::string s = "Rule of Five";
      Buffer buf(s.c_str(), s.size() + 1);
      std::cout << buf << "\n";
    }

**When to use which rule:**

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Rule
     - When to Use
     - Example
   * - Rule of Zero
     - No custom destructor or special member functions needed
     - ``std::string``, ``std::vector``, smart pointers as members
   * - Rule of Three
     - Custom destructor required (pre-C++11)
     - Raw pointers, C file handles
   * - Rule of Five
     - Custom destructor required (C++11+)
     - Same as Rule of Three, with move support

Move and Destructor
-------------------

:Source: `src/raii/move-leak <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/raii/move-leak>`_

A critical detail: moving from an object does **not** call its destructor. The
moved-from object remains alive until it goes out of scope. If the move constructor
fails to properly reset the source object's state, the destructor may attempt to
release resources that were already transferred, causing double-free bugs—or worse,
the moved-from object may still hold resources that leak.

**Problematic move constructor (causes resource leak):**

.. code-block:: cpp

    #include <cstdio>
    #include <utility>

    class File {
     public:
      explicit File(const char *path) : handle_(std::fopen(path, "w")) {}

      ~File() {
        if (handle_) std::fclose(handle_);
      }

      // Bug: does not reset other.handle_ to nullptr
      File(File &&other) noexcept : handle_(other.handle_) {
        // other.handle_ still points to the file!
      }

     private:
      std::FILE *handle_;
    };

    int main() {
      File f1("/tmp/test.txt");
      File f2(std::move(f1));
      // When f1's destructor runs, it closes the file
      // When f2's destructor runs, it closes the same file again -> double-free!
    }

**Correct move constructor:**

Always reset the moved-from object to a valid empty state. Using ``std::exchange``
makes this pattern concise and less error-prone:

.. code-block:: cpp

    // Correct: reset source to nullptr
    File(File &&other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}

The moved-from object's destructor will still run, but with ``handle_`` set to
``nullptr``, the conditional check ``if (handle_)`` prevents any invalid operation.

Using std::exchange
-------------------

:Source: `src/raii/move-exchange <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/raii/move-exchange>`_

``std::exchange`` (C++14) atomically replaces a value and returns the old one,
making it ideal for implementing move operations. It eliminates the risk of
forgetting to reset the source object and produces cleaner, more readable code.

**Complete example with std::exchange:**

.. code-block:: cpp

    #include <cstdio>
    #include <utility>

    class File {
     public:
      explicit File(const char *path) : handle_(std::fopen(path, "w")) {}

      ~File() {
        if (handle_) std::fclose(handle_);
      }

      // Move constructor using std::exchange
      File(File &&other) noexcept
          : handle_(std::exchange(other.handle_, nullptr)) {}

      // Move assignment using std::exchange
      File &operator=(File &&other) noexcept {
        if (this != &other) {
          if (handle_) std::fclose(handle_);  // Release current resource
          handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
      }

      // Non-copyable
      File(const File &) = delete;
      File &operator=(const File &) = delete;

     private:
      std::FILE *handle_ = nullptr;
    };

**Alternative: swap-based move assignment:**

Another common pattern uses ``std::swap``, which is exception-safe and handles
self-assignment automatically:

.. code-block:: cpp

    File &operator=(File &&other) noexcept {
      std::swap(handle_, other.handle_);
      return *this;
      // other's destructor will close our old handle
    }

Value Categories
----------------

:Source: `src/raii/value-categories <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/raii/value-categories>`_

Understanding value categories is essential for mastering move semantics. C++11
introduced a refined taxonomy that determines when move operations are invoked
versus copy operations. Every expression in C++ has both a type and a value
category. The value category determines whether the expression can be moved from,
whether it has identity (can take its address), and which overloaded function
will be selected during overload resolution. This classification is fundamental
to understanding why ``std::move`` works and when move constructors are called.

**The five value categories:**

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Category
     - Has Identity
     - Description
   * - lvalue
     - Yes
     - Has a name or address; can appear on left side of assignment
   * - xvalue
     - Yes
     - "eXpiring value"; about to be moved from (e.g., ``std::move(x)``)
   * - prvalue
     - No
     - "Pure rvalue"; temporary with no name (e.g., ``42``, ``Foo()``)
   * - glvalue
     - Yes
     - Generalized lvalue (lvalue or xvalue)
   * - rvalue
     - -
     - Can be moved from (xvalue or prvalue)

**What std::move actually does:**

``std::move`` is an unconditional cast to rvalue reference—it does not move
anything by itself. The actual move happens when a move constructor or assignment
is invoked. See `Common Move Semantics Pitfalls`_ for details and examples.

Move-Only Types
---------------

:Source: `src/raii/move-only <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/raii/move-only>`_

Some types represent unique ownership of a resource and should never be copied—only
moved. Examples include ``std::unique_ptr``, ``std::thread``, ``std::fstream``, and
custom file handles. These types enforce exclusive ownership semantics: at any point
in time, exactly one object owns the resource. Copying would violate this invariant
by creating two owners, leading to double-free bugs or resource conflicts. Implement
move-only types by explicitly deleting copy operations while providing move operations:

.. code-block:: cpp

    #include <cstdio>
    #include <utility>

    class UniqueFile {
     public:
      explicit UniqueFile(const char *path)
          : handle_(std::fopen(path, "w")) {}

      ~UniqueFile() {
        if (handle_) std::fclose(handle_);
      }

      // Delete copy operations
      UniqueFile(const UniqueFile &) = delete;
      UniqueFile &operator=(const UniqueFile &) = delete;

      // Provide move operations
      UniqueFile(UniqueFile &&other) noexcept
          : handle_(std::exchange(other.handle_, nullptr)) {}

      UniqueFile &operator=(UniqueFile &&other) noexcept {
        if (this != &other) {
          if (handle_) std::fclose(handle_);
          handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
      }

      std::FILE *get() const { return handle_; }

     private:
      std::FILE *handle_ = nullptr;
    };

    int main() {
      UniqueFile f1("/tmp/test.txt");
      // UniqueFile f2 = f1;           // Error: copy deleted
      UniqueFile f2 = std::move(f1);   // OK: move
    }

Perfect Forwarding
------------------

:Source: `src/raii/forwarding <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/raii/forwarding>`_

Perfect forwarding preserves the value category (lvalue or rvalue) of arguments
when passing them through template functions. Without perfect forwarding, a named
parameter inside a function is always an lvalue (because it has a name), even if
the original argument was an rvalue. This causes unnecessary copies when the
forwarded argument should have been moved. Perfect forwarding is essential for
implementing generic wrapper functions, factory functions like ``std::make_unique``,
and decorator patterns that need to forward arguments to other functions without
losing efficiency or changing semantics.

**Universal references (forwarding references):**

In template contexts, ``T&&`` is a *universal reference* (also called *forwarding
reference* in C++17 terminology), not an rvalue reference. This special deduction
rule only applies when ``T`` is a deduced template parameter. Universal references
bind to both lvalues and rvalues through reference collapsing rules, making them
the foundation of perfect forwarding:

.. code-block:: cpp

    template <typename T>
    void wrapper(T&& arg);  // Universal reference, not rvalue reference

    int x = 42;
    wrapper(x);              // T deduced as int&, T&& becomes int& (lvalue)
    wrapper(42);             // T deduced as int, T&& becomes int&& (rvalue)

**Reference collapsing rules:**

When references to references are formed (through template instantiation or
typedef), they collapse according to these rules. The key insight is that any
reference involving an lvalue reference collapses to an lvalue reference:

.. code-block:: text

    T& &   → T&
    T& &&  → T&
    T&& &  → T&
    T&& && → T&&

**Using std::forward:**

``std::forward<T>`` is a conditional cast: it casts to rvalue reference only if
``T`` is not an lvalue reference type. This preserves the original value category
of the argument through the forwarding chain. Unlike ``std::move`` which always
casts to rvalue, ``std::forward`` examines the template parameter to decide:

.. code-block:: cpp

    #include <iostream>
    #include <utility>

    void process(int &x) { std::cout << "lvalue: " << x << "\n"; }
    void process(int &&x) { std::cout << "rvalue: " << x << "\n"; }

    template <typename T>
    void wrapper(T &&arg) {
      process(std::forward<T>(arg));  // Preserves value category
    }

    int main() {
      int x = 42;
      wrapper(x);        // Calls process(int&)
      wrapper(123);      // Calls process(int&&)
    }

**Factory pattern with perfect forwarding:**

.. code-block:: cpp

    #include <memory>
    #include <utility>

    template <typename T, typename... Args>
    std::unique_ptr<T> make_unique(Args&&... args) {
      return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }

    struct Widget {
      Widget(int x, double y) : x_(x), y_(y) {}
      int x_;
      double y_;
    };

    int main() {
      auto w = make_unique<Widget>(42, 3.14);  // Perfect forwarding
    }

For more details on value categories and forwarding, see the C++ standard
documentation on expression categories.

Conditional noexcept
--------------------

:Source: `src/raii/conditional-noexcept <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/raii/conditional-noexcept>`_

Move operations should be ``noexcept`` whenever possible. This is critical for
performance because the standard library containers make optimization decisions
based on the ``noexcept`` specification. Specifically, ``std::vector`` uses move
operations during reallocation only if they are ``noexcept``; otherwise, it falls
back to copying for exception safety. The reason is that if a move operation throws
mid-reallocation, the vector would be left in an inconsistent state with some
elements moved and others not. By requiring ``noexcept``, the vector can safely
use moves knowing they won't throw.

**Why noexcept matters:**

.. code-block:: cpp

    #include <vector>

    struct Widget {
      Widget(Widget &&) noexcept { /* ... */ }  // std::vector will move
    };

    struct Gadget {
      Gadget(Gadget &&) { /* ... */ }  // std::vector will copy (not noexcept)
    };

**Conditional noexcept:**

When a class contains members, the move operation's exception specification should
depend on the members' move operations. A wrapper class should be ``noexcept`` if
and only if its contained type's move is ``noexcept``. Use ``noexcept(noexcept(...))``
or type traits to conditionally propagate exception specifications. This ensures
that your wrapper types work efficiently with standard containers when possible:

.. code-block:: cpp

    #include <type_traits>
    #include <utility>

    template <typename T>
    class Wrapper {
     public:
      Wrapper(Wrapper &&other) noexcept(std::is_nothrow_move_constructible_v<T>)
          : data_(std::move(other.data_)) {}

     private:
      T data_;
    };

**Checking move operation properties:**

The ``<type_traits>`` header provides compile-time checks for move operation
properties. Use these to verify your types meet the requirements for efficient
container operations:

.. code-block:: cpp

    #include <string>
    #include <type_traits>

    static_assert(std::is_nothrow_move_constructible_v<std::string>);
    static_assert(std::is_nothrow_move_assignable_v<std::string>);

Self-Move Assignment
--------------------

:Source: `src/raii/self-move <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/raii/self-move>`_

Self-move assignment (``x = std::move(x)``) is rare in practice but can occur
through aliasing, particularly when working with pointers or references. While
the standard library guarantees that self-move leaves objects in a valid but
unspecified state, user-defined types must explicitly handle this case to avoid
undefined behavior. A naive implementation that deletes resources before checking
for self-assignment will corrupt the object.

**Problematic implementation:**

.. code-block:: cpp

    class Buffer {
     public:
      Buffer &operator=(Buffer &&other) noexcept {
        delete[] data_;
        data_ = std::exchange(other.data_, nullptr);  // Bug if this == &other
        return *this;
      }
     private:
      char *data_;
    };

    Buffer b;
    b = std::move(b);  // Undefined behavior: data_ deleted then set to nullptr

**Correct implementations:**

**Option 1: Explicit self-check:**

The straightforward approach adds a self-assignment check before modifying state:

.. code-block:: cpp

    Buffer &operator=(Buffer &&other) noexcept {
      if (this != &other) {
        delete[] data_;
        data_ = std::exchange(other.data_, nullptr);
      }
      return *this;
    }

**Option 2: Swap-based (handles self-move automatically):**

The swap idiom handles self-move naturally because swapping an object with itself
is a no-op. This approach is also exception-safe and often results in cleaner code:

.. code-block:: cpp

    Buffer &operator=(Buffer &&other) noexcept {
      std::swap(data_, other.data_);
      return *this;
      // other's destructor cleans up our old data
    }

The swap-based approach is preferred because it handles self-move naturally
without explicit checks and is exception-safe.

Moved-From State
----------------

:Source: `src/raii/moved-from-state <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/raii/moved-from-state>`_

After an object is moved from, it must remain in a *valid but unspecified state*.
This is a crucial invariant that allows moved-from objects to be safely destroyed
or reassigned. The "valid" part means the object's class invariants are maintained
enough for the destructor to run correctly. The "unspecified" part means you cannot
rely on any particular value—the object might be empty, might retain its old value,
or might be in some other consistent state. This guarantee applies to standard
library types and should be maintained by user-defined types.

**What operations are valid on moved-from objects:**

1. The object's destructor can be safely called
2. The object can be assigned a new value
3. Other operations have unspecified behavior (don't rely on them)

**Standard library guarantees:**

.. code-block:: cpp

    #include <string>
    #include <vector>

    std::string s1 = "hello";
    std::string s2 = std::move(s1);

    // Valid operations on s1:
    s1.~basic_string();  // OK: destructor
    s1 = "world";        // OK: assignment

    // Unspecified behavior (avoid):
    // s1.size()         // Unspecified result
    // s1[0]             // Undefined behavior if empty

**Implementing moved-from state:**

For user-defined types, ensure moved-from objects are in a valid state that
can be safely destroyed. The simplest approach is to reset all resource handles
to null or zero, which the destructor already handles:

.. code-block:: cpp

    class Resource {
     public:
      Resource(Resource &&other) noexcept
          : ptr_(std::exchange(other.ptr_, nullptr)),
            size_(std::exchange(other.size_, 0)) {
        // other is now in valid empty state
      }

      ~Resource() {
        delete[] ptr_;  // Safe even if ptr_ is nullptr
      }

     private:
      int *ptr_ = nullptr;
      size_t size_ = 0;
    };

See `Common Move Semantics Pitfalls`_ for examples of incorrect usage patterns.

Return Value Optimization and Move Semantics
---------------------------------------------

When returning objects from functions, the compiler applies optimizations that
can eliminate both copy and move operations entirely. Understanding the interaction
between RVO, NRVO, and move semantics is crucial for writing efficient code. In
many cases, the compiler constructs the return value directly in the caller's
memory, bypassing both copy and move constructors. This optimization is so
important that C++17 made it mandatory for certain cases (guaranteed copy elision).

**Guaranteed copy elision (C++17):**

When returning a prvalue (temporary), C++17 guarantees copy elision—no copy or
move constructor is called. The object is constructed directly in the caller's
storage:

.. code-block:: cpp

    Widget make_widget() {
      return Widget(42);  // Guaranteed elision: constructed directly in caller
    }

**Named Return Value Optimization (NRVO):**

When returning a named local variable, NRVO is an optional optimization that most
compilers implement. However, it cannot apply in all cases (e.g., multiple return
paths with different variables). If NRVO doesn't apply, the compiler uses the move
constructor if available, otherwise falls back to copy:

.. code-block:: cpp

    Widget make_widget(bool flag) {
      Widget w1(1);
      Widget w2(2);
      return flag ? w1 : w2;  // NRVO cannot apply (multiple return paths)
                               // Falls back to move constructor
    }

**Don't use std::move on return:**

A common mistake is using ``std::move`` when returning local variables. This is
a pessimization because it prevents RVO/NRVO and forces a move operation when
elision would have eliminated the operation entirely. The compiler already treats
returned local variables as rvalues when move is needed:

.. code-block:: cpp

    // Bad: prevents RVO
    Widget make_widget() {
      Widget w(42);
      return std::move(w);  // Pessimization!
    }

    // Good: allows RVO
    Widget make_widget() {
      Widget w(42);
      return w;  // RVO or move (compiler decides)
    }

For comprehensive coverage of RVO, NRVO, and copy elision, see
:doc:`cpp_rvo`.

Emplace vs Insert
-----------------

:Source: `src/raii/emplace <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/raii/emplace>`_

Container operations like ``emplace_back`` and ``emplace`` construct elements
in-place, avoiding temporary objects and unnecessary moves/copies. This is
particularly beneficial for types that are expensive to construct.

**push_back vs emplace_back:**

.. code-block:: cpp

    #include <vector>
    #include <string>

    std::vector<std::string> v;

    // push_back: creates temporary, then moves it into vector
    v.push_back(std::string("hello"));  // 1 construction + 1 move

    // emplace_back: constructs directly in vector's storage
    v.emplace_back("hello");            // 1 construction only

**When push_back copies vs moves:**

``push_back`` copies when given an lvalue, moves when given an rvalue:

.. code-block:: cpp

    std::vector<std::string> v;
    std::string s = "hello";

    v.push_back(s);             // lvalue: COPIES s into vector
    v.push_back(std::move(s));  // rvalue: MOVES s into vector
    v.push_back("world");       // rvalue (temporary): MOVES into vector

**Expensive types benefit most:**

For types with costly constructors, avoiding the temporary + move overhead matters:

.. code-block:: cpp

    std::vector<std::vector<int>> vv;

    // emplace_back: constructs 1000-element vector directly in vv
    vv.emplace_back(1000, 0);

    // push_back equivalent would be:
    // vv.push_back(std::vector<int>(1000, 0));  // construct + move

**Map emplace avoids pair construction:**

.. code-block:: cpp

    std::map<std::string, std::string> m;

    // emplace: constructs pair<string,string> in-place
    m.emplace("key", "value");

    // insert equivalent would be:
    // m.insert(std::make_pair("key", "value"));  // construct pair + move

**Non-copyable types:**

For move-only types, ``emplace`` constructs in-place without requiring a copy:

.. code-block:: cpp

    struct NonCopyable {
      int value;
      explicit NonCopyable(int v) : value(v) {}
      NonCopyable(const NonCopyable&) = delete;
      NonCopyable(NonCopyable&&) noexcept = default;
    };

    std::vector<NonCopyable> v;
    v.emplace_back(42);  // OK: constructs in-place

**Caveat: emplace can hide bugs:**

``emplace`` uses perfect forwarding, which can lead to unexpected implicit
conversions that ``push_back`` would catch:

.. code-block:: cpp

    std::vector<std::vector<int>> vv;

    vv.push_back(10);     // Error: no conversion from int to vector<int>
    vv.emplace_back(10);  // Compiles! Creates vector<int>(10) - probably not intended

Common Move Semantics Pitfalls
-------------------------------

:Source: `src/raii/move-pitfalls <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/raii/move-pitfalls>`_

Move semantics introduce several subtle pitfalls that can lead to bugs or
performance issues. This section catalogs the most common mistakes developers
make when working with move operations.

**1. std::move doesn't move:**

The most common misconception is that ``std::move`` performs a move. It doesn't—it's
just an unconditional cast to rvalue reference. The actual move happens when a move
constructor or move assignment is invoked with the result:

.. code-block:: cpp

    std::string s1 = "hello";
    std::move(s1);              // Does nothing!
    std::string s2 = std::move(s1);  // Now s1 is moved

**2. Moving from const objects copies:**

Move operations require modifying the source object, so they cannot work with
``const`` objects. When you try to move from a ``const`` object, overload resolution
selects the copy constructor instead, silently performing a copy:

.. code-block:: cpp

    const std::string s1 = "hello";
    std::string s2 = std::move(s1);  // Calls copy constructor, not move

**3. Accidentally using moved-from objects:**

After moving from an object, it's in a valid but unspecified state. Using it
(other than destruction or assignment) leads to unspecified or undefined behavior:

.. code-block:: cpp

    std::vector<int> v1 = {1, 2, 3};
    std::vector<int> v2 = std::move(v1);

    for (int x : v1) { /* ... */ }  // Bug: v1 is in unspecified state

    v1 = {4, 5, 6};                 // OK: assign new value first
    for (int x : v1) { /* ... */ }  // Now safe

**4. Forgetting noexcept on move operations:**

Without ``noexcept``, standard containers fall back to copying during reallocation
for exception safety. This can cause significant performance degradation:

.. code-block:: cpp

    struct Widget {
      Widget(Widget &&) { /* ... */ }  // Missing noexcept
    };

    std::vector<Widget> v;
    v.push_back(Widget());
    // Vector reallocation will copy instead of move!

**5. Not resetting moved-from state:**

Failing to reset the source object's state in move operations leads to double-free
bugs when both objects' destructors run:

.. code-block:: cpp

    class Resource {
     public:
      Resource(Resource &&other) noexcept : ptr_(other.ptr_) {
        // Bug: forgot to set other.ptr_ = nullptr
      }
      ~Resource() { delete ptr_; }  // Double-free!
     private:
      int *ptr_;
    };

**6. Self-move without protection:**

Self-move through aliasing (e.g., ``x = std::move(*ptr)`` where ``ptr == &x``)
can corrupt objects if not handled. Use explicit checks or the swap idiom:

.. code-block:: cpp

    Buffer &operator=(Buffer &&other) noexcept {
      delete[] data_;
      data_ = other.data_;  // Bug if this == &other
      return *this;
    }

Preventing Object Slicing
-------------------------

:Source: `src/raii/polymorphic <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/raii/polymorphic>`_

Polymorphic base classes should suppress public copy and move operations to prevent
*object slicing*. Slicing occurs when a derived class object is copied into a base
class object, losing the derived class's data and behavior.

**Problematic code (allows slicing):**

.. code-block:: cpp

    #include <iostream>
    #include <string>

    class Animal {
     public:
      virtual std::string speak() { return "..."; }
    };

    class Dog : public Animal {
     public:
      std::string speak() override { return "woof"; }
    };

    void process(Animal &a) {
      auto copy = a;  // Slicing! Only Animal part is copied
      std::cout << copy.speak() << "\n";  // Prints "..." not "woof"
    }

    int main(int argc, char *argv[]) {
      Dog dog;
      process(dog);  // Output: "..." (sliced!)
    }

**Correct approach (prevents slicing):**

By deleting copy operations in the base class, the compiler prevents accidental
slicing at compile time:

.. code-block:: cpp

    #include <iostream>
    #include <string>

    class Animal {
     public:
      Animal() = default;
      Animal(const Animal &) = delete;
      Animal &operator=(const Animal &) = delete;
      virtual ~Animal() = default;

      virtual std::string speak() { return "..."; }
    };

    class Dog : public Animal {
     public:
      std::string speak() override { return "woof"; }
    };

    void process(Animal &a) {
      // auto copy = a;  // Compile error: copy constructor deleted
      std::cout << a.speak() << "\n";
    }

    int main(int argc, char *argv[]) {
      Dog dog;
      process(dog);  // Output: "woof"
    }

Initializer Lists
-----------------

:Source: `src/raii/initializer-list <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/raii/initializer-list>`_

``std::initializer_list`` provides a lightweight proxy for accessing an array of
objects. It enables uniform initialization syntax and is commonly used for
constructors that accept a variable number of homogeneous arguments.

The initializer list does not own its elements; it merely provides access to a
temporary array created by the compiler. This makes it efficient for passing
initialization data:

.. code-block:: cpp

    #include <initializer_list>
    #include <iostream>

    template <typename T>
    T sum(std::initializer_list<T> values) {
      T result = 0;
      for (const auto &v : values) {
        result += v;
      }
      return result;
    }

    int main(int argc, char *argv[]) {
      std::cout << sum({1, 2, 3, 4, 5}) << "\n";      // Output: 15
      std::cout << sum({1.5, 2.5, 3.0}) << "\n";      // Output: 7.0
    }

RAII Wrapper
------------

:Source: `src/raii/raii-wrapper <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/raii/raii-wrapper>`_

The RAII wrapper pattern encapsulates resource management in a class, ensuring
resources are properly acquired in the constructor and released in the destructor.
This pattern is the foundation of smart pointers, lock guards, and file handles
in the C++ standard library.

This example demonstrates a simple RAII wrapper for a file handle:

.. code-block:: cpp

    #include <cstdio>
    #include <stdexcept>
    #include <utility>

    class File {
     public:
      explicit File(const char *path, const char *mode) : handle_(std::fopen(path, mode)) {
        if (!handle_) {
          throw std::runtime_error("Failed to open file");
        }
      }

      ~File() {
        if (handle_) {
          std::fclose(handle_);
        }
      }

      // Non-copyable
      File(const File &) = delete;
      File &operator=(const File &) = delete;

      // Movable
      File(File &&other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
      File &operator=(File &&other) noexcept {
        if (this != &other) {
          if (handle_) std::fclose(handle_);
          handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
      }

      std::FILE *get() const { return handle_; }

     private:
      std::FILE *handle_;
    };

    int main(int argc, char *argv[]) {
      try {
        File f("/tmp/test.txt", "w");
        std::fputs("RAII in action\n", f.get());
      } catch (const std::exception &e) {
        // File automatically closed even if exception thrown
      }
      // File automatically closed when 'f' goes out of scope
    }

.. note::

    Modern C++ provides ``std::unique_ptr`` with custom deleters and ``std::fstream``
    for most resource management needs. Use custom RAII wrappers only when standard
    library alternatives are insufficient.
