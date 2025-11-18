=========
coroutine
=========

.. contents:: Table of Contents
    :backlinks: none

Generator
---------

.. code-block:: cpp

    // g++ -O3 -std=c++20 -Wall -Werror co.cc

    #include <iostream>
    #include <coroutine>

    template <typename T>
    class generator {
    public:
      struct promise_type;
      using handle_type = std::coroutine_handle<promise_type>;
      handle_type h_;

      struct promise_type {
        T value_;
        std::exception_ptr exception_;
        generator<T> get_return_object() {
          return { handle_type::from_promise(*this) };
        }
        void unhandled_exception() { exception_ = std::current_exception(); }
        void return_void() {}
        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        std::suspend_always yield_value(T v) noexcept {
          value_ = std::move(v);
          return {};
        }
      };

    public:

      generator(handle_type h) : h_(h) {}
      ~generator() { h_.destroy(); }
      explicit operator bool() {
        next();
        return !h_.done();
      }

      T operator() () {
        next();
        cached_ = false;
        return std::move(h_.promise().value_);
      }

    private:
      bool cached_ = false;
      void next() {
        if (cached_) {
          return;
        }
        h_();
        if (h_.promise().exception_) {
          std::rethrow_exception(h_.promise().exception_);
        }
        cached_ = true;
      }
    };

    generator<uint64_t> fib(uint64_t n) {
      uint64_t a = 0, b = 1;
      for (uint64_t i = 0; i <= n; ++i) {
        co_yield a;
        uint64_t t = b;
        b = a + b;
        a = t;
      }
    }

    int main(int argc, char *argv[]) {
      auto g = fib(10);
      while (g) {
        std::cout << g() << " ";
      }
      // ./a.out
      //  0 1 1 2 3 5 8 13 21 34 55
    }

Boost ASIO Echo Server
----------------------

.. code-block:: cpp

    #include <iostream>
    #include <boost/asio/co_spawn.hpp>
    #include <boost/asio/detached.hpp>
    #include <boost/asio/io_context.hpp>
    #include <boost/asio/ip/tcp.hpp>
    #include <boost/asio/signal_set.hpp>
    #include <boost/asio/write.hpp>

    using boost::asio::ip::tcp;
    using boost::asio::awaitable;
    using boost::asio::co_spawn;
    using boost::asio::detached;
    using boost::asio::use_awaitable;
    namespace this_coro = boost::asio::this_coro;

    constexpr uint64_t BUFSIZE = 1024;

    awaitable<void> echo(tcp::socket &socket) {
      for (;;) {
        char data[BUFSIZE] = {0};
        auto n = co_await socket.async_read_some(boost::asio::buffer(data), use_awaitable);
        co_await async_write(socket, boost::asio::buffer(data, n), use_awaitable);
      }
    }

    awaitable<void> handle(tcp::socket socket) {
      try {
        co_await echo(socket);
      } catch(const std::exception &e) {
        std::cerr << e.what();
      }
    }

    awaitable<void> listener() {
      auto e = co_await this_coro::executor;
      tcp::acceptor acceptor(e, {tcp::v4(), 8888});
      for (;;) {
        tcp::socket socket = co_await acceptor.async_accept(use_awaitable);
        co_spawn(e, handle(std::move(socket)), detached);
      }
    }

    int main(int argc, char *argv[]) {
      boost::asio::io_context io_context;
      boost::asio::signal_set signals(io_context, SIGINT, SIGTERM);
      signals.async_wait([&](auto, auto){ io_context.stop(); });
      co_spawn(io_context, listener(), detached);
      io_context.run();
    }

.. code-block:: cmake

    # CMakeLists.txt
    cmake_minimum_required(VERSION 3.10)
    set(target a.out)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
    project(example)
    find_package(Boost)
    add_executable(${target} a.cc)
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_include_directories(${target} PRIVATE "${Boost_INCLUDE_DIR}")
    target_link_libraries(${target} ${Boost_LIBRARIES})
    target_link_libraries(${target} INTERFACE Boost::coroutine)
