=======================
C++ Template cheatsheet
=======================

Template Specialization
-----------------------

.. code-block:: cpp

    #include <iostream>

    template <typename T, typename U>
    class Base
    {
    private:
        T m_a;
        U m_b;

    public:

        Base(T a, U b) : m_a(a), m_b(b) {};

        T foo() { return m_a; }
        U bar() { return m_b; }
    };

    // partial specialization
    template<typename T>
    class Base <T, int>
    {
    private:
        T m_a;
        int m_b;
    public:
        Base(T a, int b) : m_a(a), m_b(b) {}
        T foo() { return m_a; }
        int bar() { return m_b; }
    };

    // full specialization
    template<>
    class Base <double, double>
    {
    private:
        double d_a;
        double d_b;
    public:
        Base(double a, double b) : d_a(a), d_b(b) {}
        double foo() { return d_a; }
        double bar() { return d_b; }
    };


    int main (int argc, char *argv[])
    {
        Base<float, int> foo(3.33, 1);
        Base<double, double> bar(55.66, 95.27);
        std::cout << foo.foo() << std::endl;
        std::cout << foo.bar() << std::endl;
        std::cout << bar.foo() << std::endl;
        std::cout << bar.bar() << std::endl;
        return 0;
    }

Class Template
--------------

.. code-block:: cpp

    #include <iostream>

    template <typename T>
    class Area
    {
    protected:
        T w;
        T h;
    public:
        Area(T a, T b) : w(a), h(b) {}
        T get() { return w * h; }
    };

    class Rectangle : public Area<int>
    {
    public:
        Rectangle(int a, int b) : Area<int>(a, b) {}
    };

    template <typename T>
    class GenericRectangle : public Area<T>
    {
    public:
        GenericRectangle(T a, T b) : Area<T>(a, b){}
    };


    int main (int argc, char *argv[])
    {
        Rectangle r(2, 5);
        GenericRectangle<double> g1(2.5, 3.);
        GenericRectangle<int> g2(2, 3);

        std::cout << r.get() << std::endl;
        std::cout << g1.get() << std::endl;
        std::cout << g2.get() << std::endl;
        return 0;
    }


Curiously recurring template pattern
------------------------------------

.. code-block:: cpp

    #include <iostream>

    // Curiously Recurring Template Pattern (CRTP)

    template <typename D>
    class Base
    {
    public:
        void interface() {
            static_cast<D *>(this)->implement();
        }

        static void static_interface() {
            D::static_interface();
        }

        void implement() {
            std::cout << "Base" << std::endl;
        }
    };

    class DerivedFoo : public Base<DerivedFoo>
    {
    public:
        void implement() {
            std::cout << "Foo" << std::endl;
        }
        static void static_interface() {
            std::cout << "Static Foo" << std::endl;
        }
    };

    class DerivedBar : public Base<DerivedBar> {};

    int main (int argc, char *argv[])
    {
        DerivedFoo foo;
        DerivedBar bar;

        foo.interface();
        foo.static_interface();
        bar.interface();

        return 0;
    }
