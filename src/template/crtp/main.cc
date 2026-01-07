#include <iostream>
#include <memory>

// Virtual function (dynamic polymorphism)
class VBase {
 public:
  virtual void impl() { std::cout << "VBase\n"; }
  virtual ~VBase() = default;
};

class VDerived : public VBase {
 public:
  void impl() override { std::cout << "VDerived\n"; }
};

// CRTP (static polymorphism)
template <typename D>
class Base {
 public:
  void interface() { static_cast<D*>(this)->impl(); }
  void impl() { std::cout << "Base\n"; }
};

class Derived : public Base<Derived> {
 public:
  void impl() { std::cout << "Derived\n"; }
};

int main() {
  // dynamic polymorphism
  std::unique_ptr<VBase> p = std::make_unique<VDerived>();
  p->impl();

  // static polymorphism
  Derived d;
  d.interface();
}
