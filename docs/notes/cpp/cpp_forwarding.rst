==================
Perfect Forwarding
==================

.. contents:: Table of Contents
    :backlinks: none

Perfect forwarding is a way for a programmer to design a generic wrapper
function in their C++ programs. However, few examples show how to use it in a
real scenario. Instead of explaining what a C++ `perfect forwarding` is, this
article tries to collect use cases about using it.

Decorator Pattern
-----------------

.. code-block:: cpp

    #include <iostream>
    #include <utility>
    #include <chrono>

    template <typename Func, typename ...Args>
    auto Decorator(Func &&f, Args&&... args) {

        const auto s = std::chrono::system_clock::now();
        auto ret = f(std::forward<Args>(args)...);
        const auto e = std::chrono::system_clock::now();
        std::chrono::duration<double> d = e - s;

        std::cout << "Time Cost: " << d.count() << std::endl;
        return ret;
    }

    long fib(long n) {
        return n < 2 ? 1 : fib(n-1) + fib(n-2);
    }

    int main() {
        Decorator(fib, 35);
        return 0;
    }

Profiling
---------

.. code-block:: cpp

    #include <iostream>
    #include <utility>
    #include <chrono>

    class Timer {
    public:
        Timer() : s_(std::chrono::system_clock::now()) {}
        ~Timer() {
            e_ = std::chrono::system_clock::now();
            std::chrono::duration<double> d = e_ - s_;
            std::cout << "Time Cost: " << d.count() << std::endl;
        }
    private:
        std::chrono::time_point<std::chrono::system_clock> s_;
        std::chrono::time_point<std::chrono::system_clock> e_;
    };

    template <typename Func, typename ...Args>
    auto Profile(Func f, Args&&... args) {
        Timer timer;
        return f(std::forward<Args>(args)...);
    }

    long fib1(long n) {
        return (n < 2) ? 1 : fib1(n-1) + fib1(n-2);
    }

    template<long N>
    struct f {
        static constexpr long v = f<N-1>::v + f<N-2>::v;
    };

    template<>
    struct f<0> {
        static constexpr long v = 0;
    };

    template<>
    struct f<1> {
        static constexpr long v = 1;
    };

    int main() {
        long ret = -1;
        ret = Profile(fib1, 35);
        std::cout << ret << std::endl;

        ret = Profile([](){ return f<35>::v; });
        std::cout << ret << std::endl;
        return 0;
    }

Factory Pattern
---------------

.. code-block:: cpp

    #include <iostream>
    #include <utility>
    #include <string>
    #include <memory>

    struct PostgresqlConfig { /* implementation */ };
    struct MysqlConfig { /* implementation */ };

    template <typename DB>
    class Session {
    public:
        void connect(const std::string url) {
            static_cast<DB*>(this)->connect(url);
        }
    };

    class Postgresql : public Session<Postgresql> {
    private:
        PostgresqlConfig config_;
    public:
        Postgresql(PostgresqlConfig c) : config_(c) {}

        void connect(const std::string url) {
            std::cout << "Connecting to Postgresql..." << std::endl;
            // connecting
        }
    };

    class Mysql : public Session<Mysql> {
    private:
        MysqlConfig config_;
    public:
        Mysql(MysqlConfig c) : config_(c) {}

        void connect(const std::string url) {
            std::cout << "Connecting to Mysql..." << std::endl;
            // connecting
        }
    };

    /**
     * An example of Perfect Forwarding
     */
    template <typename S, typename C>
    std::shared_ptr<S> SessionFactory(C&& c) {
        return std::make_shared<S>(std::forward<C>(c));
    }

    using PostgresSession = Session<Postgresql>;
    using MysqlSession = Session<Mysql>;
    using PostgresPtr = std::shared_ptr<PostgresSession>;
    using MysqlPtr = std::shared_ptr<MysqlSession>;

    int main(int argc, char *argv[]) {

        PostgresqlConfig pc;
        MysqlConfig mc;

        PostgresPtr ps = SessionFactory<Postgresql>(pc);
        MysqlPtr ms = SessionFactory<Mysql>(mc);

        ps->connect("postgresql://...");
        ms->connect("mysql://...");
        return 0;
    }
