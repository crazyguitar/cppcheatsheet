===========
constructor
===========

.. contents:: Table of Contents
    :backlinks: none

Constructors
------------

.. code-block:: cpp

    #include <iostream>
    #include <utility>

    class C {
     public:
      // constructor
      C(int x) : x_(x) {}

      // default constructor
      C() = default;

      // copy constructor
      C(const C &other) : C(other.x_) {
        std::cout << "copy constructor\n";
      }

      // copy assignment
      C &operator=(const C &other) {
        std::cout << "copy assignment\n";
        x_ = other.x_;
        return *this;
      }

      // move constructor
      C(C &&other) : x_(std::move(other.x_)) {
        std::cout << "move constructor\n";
        other.x_ = 0;
      }

      // move assignment
      C &operator=(C &&other) {
        std::cout << "move assignment\n";
        x_ = std::move(other.x_);
        return *this;
      }

     private:
      int x_;
    };

    int main(int argc, char *argv[]) {
      C c1;                   // call default constructor
      C c2(1);                // call constructor
      C c3 = C(2);            // call constructor
      C c4(c2);               // call copy constructor
      C c5(std::move(C(2)));  // call move constructor
      C c6 = c1;              // call copy constructor
      C c7 = std::move(C(2)); // call move constructor
      C c8 = std::move(c3);   // call move constructor

      C c9;
      C c10;

      c9 = c2;                // call copy assignment
      c10 = std::move(c4);    // call move assignment
      c10 = C(2);             // call move assignment
    }

Rule of three
-------------

.. code-block:: cpp

    #include <iostream>
    #include <memory>
    #include <string>
    #include <cstring>

    class RuleOfThree {
     public:
      RuleOfThree(const char *s, size_t n)
       : cstr_(new char[n])
       , n_(n) {
        memcpy(cstr_, s, n);
      }

      // if we have a user-defined destructor
      ~RuleOfThree() { delete[] cstr_; }
      // we need one a user-defined copy constructor
      RuleOfThree(const RuleOfThree &other)
        : RuleOfThree(other.cstr_, other.n_) {}
      // and user-defined copy assignment
      RuleOfThree &operator=(const RuleOfThree &other) {
        if (this == std::addressof(other)) {
          return *this;
        }
        delete[] cstr_;
        n_ = other.n_;
        cstr_ = new char[other.n_];
        memcpy(cstr_, other.cstr_, n_);
        return *this;
      }

      friend std::ostream &operator<<(std::ostream &os, const RuleOfThree &);

     private:
      char *cstr_;
      size_t n_;
    };

    std::ostream &operator<<(std::ostream &os, const RuleOfThree &r) {
      return os << r.cstr_;
    }

    int main(int argc, char *argv[]) {
      std::string s = "Rule of three";
      RuleOfThree r3(s.c_str(), s.size() + 1);
      std::cout << r3 << "\n";
    }

Rule of five
------------

.. code-block:: cpp

    #include <iostream>
    #include <memory>
    #include <string>
    #include <cstring>
    #include <utility>

    class RuleOfFive {
     public:
      RuleOfFive(const char *s, int n) : cstr_(new char[n]) {
        std::memcpy(cstr_, s, n);
      }

      // if there is a user-defined destructor including default or delete
      ~RuleOfFive() { delete[] cstr_; }
      // a user-defined copy constructor
      RuleOfFive(const RuleOfFive &other)
        : RuleOfFive(other.cstr_, strlen(other.cstr_) + 1) {}
      // a user-defined move constructor
      RuleOfFive(RuleOfFive &&other)
        : cstr_(std::exchange(other.cstr_, nullptr)) {}
      // a user-define copy assignment
      RuleOfFive &operator=(const RuleOfFive &other) {
        return *this = RuleOfFive(other);
      }
      // a user-defined move assignment have to declare explicitly.
      RuleOfFive &operator=(RuleOfFive &&other) {
        std::swap(cstr_, other.cstr_);
        return *this;
      }

      friend std::ostream &operator<<(std::ostream &os, const RuleOfFive &);

     private:
      char *cstr_;
    };

    std::ostream &operator<<(std::ostream &os, const RuleOfFive &r5) {
      return os << r5.cstr_;
    }

    int main(int argc, char *argv[]) {
      std::string s = "Rule of five";
      RuleOfFive r5(s.c_str(), s.size() + 1);
      std::cout << r5 << "\n";
    }

Rule of zero
------------

.. code-block:: cpp

    #include <iostream>
    #include <string>

    class RuleOfZero {
     public:
      RuleOfZero(const std::string &s) : s_(s) {}
      // if we don't have a user-defined destructor, we should not have
      // user-defined copy/move constructors or copy/move assignment.
      friend std::ostream &operator<<(std::ostream &os, const RuleOfZero &r0);
     private:
      const std::string s_;
    };

    std::ostream &operator<<(std::ostream &os, const RuleOfZero &r0) {
      return os << r0.s_;
    }

    int main(int argc, char *argv[]) {
      RuleOfZero r0("Rule of zero");
      std::cout << r0 << "\n";
    }

Note that a polymorphic class should supress public copy/move.

.. code-block:: cpp

    #include <iostream>
    #include <string>
    #include <utility>

    // bad
    class A {
     public:
      virtual std::string f() { return "a"; }
    };

    class B : public A {
     public:
      std::string f() override { return "b"; }
    };

    void func(A &a) {
      auto c = a;
      std::cout << c.f() << "\n";
    }

    int main(int argc, char *argv[]) {
      B b;
      func(b);
    }

.. code-block:: cpp

    #include <iostream>
    #include <string>
    #include <utility>

    class A {
     public:
      A() = default;
      A(const A&) = delete;
      A &operator=(const A&) = delete;
      virtual std::string f() { return "a"; }
    };

    class B : public A {
     public:
      std::string f() override { return "b"; }
    };

    void func(A &a) {
      auto c = a;  // compile error here!
      std::cout << c.f() << "\n";
    }

    int main(int argc, char *argv[]) {
      B b;
      func(b);
    }
