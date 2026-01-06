=========
Coroutine
=========

.. meta::
   :description: C++20 coroutine tutorial covering co_await, co_yield, co_return, promise types, awaiters, generators, and async I/O with practical examples.
   :keywords: C++, C++20, coroutine, co_await, co_yield, co_return, async, asynchronous, promise type, awaiter, generator, tutorial

.. contents:: Table of Contents
    :backlinks: none

Introduction
------------

C++20 coroutines provide a powerful mechanism for writing asynchronous code that
maintains the readability of synchronous code. Unlike traditional functions that
execute from start to finish in a single invocation, coroutines can **suspend**
their execution at specific points and **resume** later, potentially after some
external event occurs (like I/O completion or a timer expiring). This capability
enables efficient cooperative multitasking without the complexity of explicit
callbacks, state machines, or thread synchronization.

When the compiler encounters a function containing ``co_await``, ``co_yield``, or
``co_return``, it transforms that function into a coroutine. The transformation
creates a state machine that tracks the coroutine's progress through its execution
points. The compiler also generates a **coroutine frame** (typically heap-allocated)
that stores local variables across suspension points, and a **promise object** that
controls the coroutine's behavior at each suspension and resumption.

.. list-table:: Coroutine Keywords
   :header-rows: 1
   :widths: 20 80

   * - Keyword
     - Description
   * - ``co_await``
     - Suspend execution until the awaited operation completes; the awaited
       expression determines whether suspension actually occurs and what value
       is produced when resuming
   * - ``co_yield``
     - Suspend and produce a value to the caller; syntactic sugar for
       ``co_await promise.yield_value(expr)``; commonly used in generators
   * - ``co_return``
     - Complete the coroutine and optionally return a final value; after this,
       the coroutine cannot be resumed

Coroutine Basics
----------------

:Source: `src/coroutine/basics <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/coroutine/basics>`_

A minimal coroutine requires a return type with a nested ``promise_type`` that
satisfies the coroutine promise requirements. The compiler uses this promise type
to manage the coroutine's lifecycle, including how it starts, how it handles
return values and exceptions, and what happens when it completes. The examples
use the io library's ``Coro<T>`` type which provides a complete promise
implementation with scheduling support.

.. code-block:: cpp

    #include <io/coro.h>
    #include <io/future.h>
    #include <io/io.h>

    Coro<> hello() {
      // do work
      co_return;
    }

    int main() {
      auto fut = Future(hello());  // Schedule coroutine
      IO::Get().Run();             // Run event loop
    }

Promise Type
------------

:Source: `src/coroutine/promise-type <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/coroutine/promise-type>`_

The promise type is the central control mechanism for a coroutine. It defines the
coroutine's behavior at every significant point in its lifecycle: startup,
suspension, resumption, completion, and error handling. The compiler generates
code that calls these methods at the appropriate times, allowing you to customize
how your coroutine type behaves. Understanding the promise type is essential for
building custom coroutine types.

.. list-table:: Promise Type Methods
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Description
   * - ``get_return_object()``
     - Called immediately after promise construction to create the object
       returned to the caller; happens before the coroutine body executes
   * - ``initial_suspend()``
     - Return ``suspend_always`` for lazy coroutines that wait for explicit
       resumption, or ``suspend_never`` for eager coroutines that start immediately
   * - ``final_suspend()``
     - Called after coroutine completes; must be ``noexcept``; returning
       ``suspend_always`` keeps the frame alive for result retrieval
   * - ``return_void()`` / ``return_value(v)``
     - Called on ``co_return`` or ``co_return expr``; use ``return_void()`` for
       coroutines that don't produce a value
   * - ``unhandled_exception()``
     - Called when an exception escapes the coroutine body; typically stores
       the exception via ``std::current_exception()`` for later rethrowing
   * - ``yield_value(v)``
     - Called on ``co_yield expr``; returns an awaiter that controls suspension;
       used primarily in generator patterns

.. code-block:: cpp

    #include <io/coro.h>
    #include <io/future.h>
    #include <io/io.h>

    Coro<int> compute() {
      co_return 42;
    }

    int main() {
      auto fut = Future(compute());
      IO::Get().Run();
      assert(fut.result() == 42);
    }

Awaiter
-------

:Source: `src/coroutine/awaiter <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/coroutine/awaiter>`_

An awaiter is an object that controls what happens at a ``co_await`` expression.
When you write ``co_await expr``, the compiler obtains an awaiter from the
expression (either directly if it has the right interface, or via
``operator co_await``). The awaiter then determines whether the coroutine
actually suspends, what happens during suspension, and what value the
``co_await`` expression produces. The three awaiter methods form a protocol:

- ``await_ready()``: Called first; if it returns ``true``, the coroutine skips
  suspension entirely (the result is already available)
- ``await_suspend(handle)``: Called only if ``await_ready()`` returned ``false``;
  receives the coroutine handle and can schedule resumption; can return ``void``
  (always suspend), ``bool`` (suspend if true), or another handle (symmetric transfer)
- ``await_resume()``: Called when the coroutine resumes; its return value becomes
  the result of the ``co_await`` expression

.. code-block:: cpp

    struct ReturnValue {
      int value;
      bool await_ready() const noexcept { return true; }
      void await_suspend(std::coroutine_handle<>) const noexcept {}
      int await_resume() const noexcept { return value; }
    };

    Coro<> example() {
      int x = co_await ReturnValue{42};  // x = 42, no suspension
    }

Generator
---------

:Source: `src/coroutine/generator <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/coroutine/generator>`_

A generator is a coroutine that produces a sequence of values on demand using
``co_yield``. Each ``co_yield`` expression suspends the coroutine and makes a
value available to the caller. When the caller requests the next value, the
coroutine resumes from where it left off. This pattern is ideal for lazy
sequences where values are computed only when needed, infinite streams, and
memory-efficient iteration over large datasets. Generators are a separate
pattern from async coroutines—they don't use the io library's event loop.

.. code-block:: cpp

    template <typename T>
    class Generator {
     public:
      struct promise_type {
        T current_value;
        Generator get_return_object() {
          return Generator{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        std::suspend_always yield_value(T value) {
          current_value = std::move(value);
          return {};
        }
        void return_void() {}
        void unhandled_exception() {}
      };

      struct iterator {
        std::coroutine_handle<promise_type> handle;
        iterator& operator++() { handle.resume(); return *this; }
        T& operator*() { return handle.promise().current_value; }
        bool operator==(std::default_sentinel_t) const { return handle.done(); }
      };

      iterator begin() { handle_.resume(); return {handle_}; }
      std::default_sentinel_t end() { return {}; }

      explicit Generator(std::coroutine_handle<promise_type> h) : handle_(h) {}
      ~Generator() { if (handle_) handle_.destroy(); }

     private:
      std::coroutine_handle<promise_type> handle_;
    };

    Generator<int> range(int start, int end) {
      for (int i = start; i < end; ++i) co_yield i;
    }

    int main() {
      for (int x : range(1, 5)) std::cout << x << " ";  // 1 2 3 4
    }

Asynchronous I/O
----------------

:Source: `src/coroutine/io <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/coroutine/io>`_

Traditional blocking I/O wastes CPU cycles waiting for operations to complete.
Thread-per-connection models add overhead from context switching and memory
usage. Asynchronous I/O solves this by allowing a single thread to handle
thousands of concurrent connections through event-driven programming.

The io library combines C++20 coroutines with OS-level I/O multiplexing (epoll
on Linux, kqueue on macOS/BSD) to provide an async framework where coroutines
suspend on I/O operations and resume when data is ready. This enables writing
sequential-looking code that executes asynchronously without callback hell or
explicit state machines.

Core components:

- `Coro<T> <https://github.com/crazyguitar/cppcheatsheet/blob/master/src/coroutine/io/coro.h>`_: Coroutine type for async operations
- `Future<C> <https://github.com/crazyguitar/cppcheatsheet/blob/master/src/coroutine/io/future.h>`_: Wrapper that schedules a coroutine and provides ``result()`` access
- `IO <https://github.com/crazyguitar/cppcheatsheet/blob/master/src/coroutine/io/io.h>`_: Event loop with ready queue, delayed task queue, and I/O selector integration
- `Stream <https://github.com/crazyguitar/cppcheatsheet/blob/master/src/coroutine/io/stream.h>`_: Non-blocking socket I/O with async Read/Write operations

Coroutine
~~~~~~~~~

:Source: `src/coroutine/coro <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/coroutine/coro>`_

Coroutines can be chained using ``co_await``—when you await another ``Coro<T>``,
the current coroutine suspends and automatically resumes when the awaited
coroutine completes. The inner coroutine stores a pointer to the outer coroutine's
handle, and when it finishes, it schedules the outer coroutine to resume. This
enables composing complex async operations from simpler building blocks while
maintaining readable, sequential-looking code.

.. code-block:: cpp

    #include <io/coro.h>
    #include <io/future.h>
    #include <io/io.h>

    Coro<int> square(int x) { co_return x * x; }

    Coro<int> sum_of_squares(int a, int b) {
      auto a2 = co_await square(a);
      auto b2 = co_await square(b);
      co_return a2 + b2;
    }

    template <typename C>
    decltype(auto) Run(C&& coro) {
      auto fut = Future(std::move(coro));
      IO::Get().Run();
      return std::move(fut).result();
    }

    int main() {
      assert(Run(sum_of_squares(3, 4)) == 25);
    }

Future
~~~~~~

:Source: `src/coroutine/future <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/coroutine/future>`_

The ``Future`` wrapper automatically schedules a coroutine upon construction and
provides access to its result. When you create a ``Future``, it immediately adds
the coroutine to the scheduler's ready queue. After calling ``IO::Get().Run()``,
you can retrieve the return value or any exception via ``result()``.

.. code-block:: cpp

    #include <io/coro.h>
    #include <io/future.h>
    #include <io/io.h>

    Coro<int> compute() { co_return 42; }

    int main() {
      auto fut = Future(compute());  // Schedules coroutine
      IO::Get().Run();               // Run until complete
      assert(fut.result() == 42);    // Get result
    }

IO (Scheduler)
~~~~~~~~~~~~~~

:Source: `src/coroutine/scheduler <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/coroutine/scheduler>`_

The ``IO`` class provides an event loop that orchestrates coroutine execution.
It maintains a ready queue for tasks that can run immediately, a priority queue
for delayed tasks (sorted by wake time), and integrates with the platform's I/O
selector (epoll/kqueue) for socket readiness events. The event loop processes
ready tasks, checks for I/O events, and advances delayed tasks whose timers have
expired. The ``Sleep`` awaiter uses the delayed task queue to suspend coroutines
for a specified duration.

.. code-block:: cpp

    #include <io/handle.h>
    #include <io/io.h>
    #include <io/sleep.h>

    struct MyTask : Handle {
      void run() override { /* execute */ }
      void stop() override { /* cleanup */ }
    };

    MyTask task;
    IO::Get().Call(task);                              // Schedule immediately
    IO::Get().Call(std::chrono::milliseconds(100), task);  // Delayed
    IO::Get().Cancel(task);                            // Cancel
    IO::Get().Run();                                   // Run event loop

    // Sleep example
    Coro<> delayed_task() {
      co_await Sleep(std::chrono::milliseconds(100));
      // Continues after 100ms
    }

Stream
~~~~~~

:Source: `src/coroutine/stream <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/coroutine/stream>`_

The ``Stream`` class provides non-blocking socket I/O with coroutine-based async
operations. When you call ``co_await stream.Read()``, the coroutine suspends and
registers with the I/O selector; when data arrives, the selector notifies the
scheduler which resumes the coroutine. This allows a single thread to handle
many concurrent connections efficiently.

.. code-block:: cpp

    #include <io/stream.h>
    #include <io/client.h>
    #include <io/server.h>

    Coro<> handle_client(Stream stream) {
      char buf[1024];
      size_t n = co_await stream.Read(buf, sizeof(buf));
      if (n > 0) {
        co_await stream.Write(buf, n);  // Echo back
      }
    }

    Coro<> run_server(int port) {
      auto server = Server("127.0.0.1", port, handle_client);
      server.Start();
      co_await server.Wait();
    }

    Coro<> run_client(int port) {
      Client client("127.0.0.1", port);
      auto stream = co_await client.Connect();
      co_await stream.Write("hello", 5);
      char buf[256];
      size_t n = co_await stream.Read(buf, sizeof(buf));
      // buf contains "hello"
    }

The echo-server example demonstrates more advanced patterns including large data
transfer (10MB) and concurrent multi-client scenarios. See
`src/coroutine/echo-server <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/coroutine/echo-server>`_.
