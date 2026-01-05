===================
Resource Management
===================

.. contents:: Table of Contents
    :backlinks: none

Resource Acquisition Is Initialization (RAII) is a fundamental C++ programming idiom
that ties resource management to object lifetime. When an object is constructed, it
acquires resources; when it is destroyed, it releases them. This pattern eliminates
resource leaks and ensures exception safety by leveraging C++'s deterministic
destruction guarantees.

Note that `Rust <https://www.rust-lang.org/>`_ enforces ownership at compile time, offering stronger safety guarantees
than garbage-collected languages. Interestingly, Rust's ownership model closely
resembles C++'s RAII and move semantics (available since C++11). For example, Rust's
ownership transfer is analogous to ``std::move``, and its ``Drop`` trait mirrors C++
destructors. This section explores these resource management techniques in depth.


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
