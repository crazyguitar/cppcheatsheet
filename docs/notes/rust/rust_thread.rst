=======
Threads
=======

.. meta::
   :description: Rust threading compared to C++. Covers std::thread, Arc, Mutex, channels, and Send/Sync traits.
   :keywords: Rust, threads, concurrency, Arc, Mutex, channels, Send, Sync, C++ threading

.. contents:: Table of Contents
    :backlinks: none

:Source: `src/rust/threads <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/threads>`_

Rust's ownership system extends to concurrency, preventing data races at compile time.
The ``Send`` and ``Sync`` marker traits tell the compiler which types can safely cross
thread boundaries. Unlike C++ where data races are undefined behavior that might silently
corrupt data, Rust makes concurrent access errors into compile-time errors. This means
if your Rust code compiles, it's free from data races - a guarantee no other mainstream
systems language provides.

Basic Threading
---------------

Spawning threads in Rust is similar to C++, but with an important difference: the closure
passed to ``thread::spawn`` must have a ``'static`` lifetime, meaning it can't borrow
data from the parent thread (unless using scoped threads). This prevents dangling
references when the parent thread exits before the child. The ``join`` method returns
a ``Result`` because the thread might have panicked:

**C++:**

.. code-block:: cpp

    #include <thread>
    #include <iostream>

    int main() {
      std::thread t([]() {
        std::cout << "Hello from thread\n";
      });
      t.join();
    }

**Rust:**

.. code-block:: rust

    use std::thread;

    fn main() {
        let handle = thread::spawn(|| {
            println!("Hello from thread");
        });
        handle.join().unwrap();
    }

Moving Data to Threads
----------------------

When a thread needs to own data, use the ``move`` keyword to transfer ownership into
the closure. This is safer than C++ where you might accidentally capture by reference
and create a data race. For ``Copy`` types, ``move`` creates a copy; for owned types
like ``Vec`` or ``String``, it transfers ownership completely - the original variable
becomes invalid:

**C++:**

.. code-block:: cpp

    #include <thread>
    #include <vector>

    int main() {
      std::vector<int> data = {1, 2, 3};

      std::thread t([data = std::move(data)]() {
        for (int x : data) {
          std::cout << x << " ";
        }
      });
      t.join();
    }

**Rust:**

.. code-block:: rust

    use std::thread;

    fn main() {
        let data = vec![1, 2, 3];

        let handle = thread::spawn(move || {
            for x in &data {
                print!("{} ", x);
            }
        });
        handle.join().unwrap();
    }

Shared State with Arc and Mutex
-------------------------------

When multiple threads need to access the same data, use ``Arc`` (atomic reference
counting) for shared ownership and ``Mutex`` for mutual exclusion. Unlike C++ where
you can forget to lock a mutex before accessing shared data, Rust's ``Mutex`` wraps
the data it protects - you literally cannot access the data without going through
the lock. The lock guard automatically releases the lock when dropped:

**C++:**

.. code-block:: cpp

    #include <thread>
    #include <mutex>
    #include <memory>
    #include <vector>

    int main() {
      auto counter = std::make_shared<int>(0);
      std::mutex mtx;
      std::vector<std::thread> threads;

      for (int i = 0; i < 10; i++) {
        threads.emplace_back([&mtx, counter]() {
          std::lock_guard<std::mutex> lock(mtx);
          (*counter)++;
        });
      }

      for (auto& t : threads) t.join();
      std::cout << *counter;  // 10
    }

**Rust:**

.. code-block:: rust

    use std::sync::{Arc, Mutex};
    use std::thread;

    fn main() {
        let counter = Arc::new(Mutex::new(0));
        let mut handles = vec![];

        for _ in 0..10 {
            let counter = Arc::clone(&counter);
            let handle = thread::spawn(move || {
                let mut num = counter.lock().unwrap();
                *num += 1;
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        println!("{}", *counter.lock().unwrap());  // 10
    }

RwLock (Read-Write Lock)
~~~~~~~~~~~~~~~~~~~~~~~~

``RwLock`` allows multiple readers or a single writer, providing better performance
than ``Mutex`` when reads are more common than writes. Multiple threads can hold read
locks simultaneously, but a write lock requires exclusive access. This is useful for
data that's read frequently but updated rarely, like configuration or caches:

.. code-block:: rust

    use std::sync::{Arc, RwLock};
    use std::thread;

    fn main() {
        let data = Arc::new(RwLock::new(vec![1, 2, 3]));

        // Multiple readers
        let data1 = Arc::clone(&data);
        let r1 = thread::spawn(move || {
            let read = data1.read().unwrap();
            println!("Reader 1: {:?}", *read);
        });

        // Single writer
        let data2 = Arc::clone(&data);
        let w = thread::spawn(move || {
            let mut write = data2.write().unwrap();
            write.push(4);
        });

        r1.join().unwrap();
        w.join().unwrap();
    }

Channels (Message Passing)
--------------------------

Channels provide a way for threads to communicate by sending messages rather than
sharing memory. Rust's standard library provides ``mpsc`` (multi-producer, single-consumer)
channels. The sender can be cloned to allow multiple producers, but there's only one
receiver. Channels transfer ownership of sent values, preventing data races by design.
This is Rust's preferred concurrency model - "share memory by communicating" rather
than "communicate by sharing memory":

**C++:**

.. code-block:: cpp

    // No standard channel, typically use condition variables
    // or third-party libraries

**Rust:**

.. code-block:: rust

    use std::sync::mpsc;  // multi-producer, single-consumer
    use std::thread;

    fn main() {
        let (tx, rx) = mpsc::channel();

        thread::spawn(move || {
            tx.send("hello").unwrap();
        });

        let msg = rx.recv().unwrap();
        println!("{}", msg);
    }

Multiple Producers
~~~~~~~~~~~~~~~~~~

Clone the sender to create multiple producers. Each clone can be moved to a different
thread. The receiver iterates over messages until all senders are dropped. This pattern
is useful for worker pools where multiple threads produce results that a single thread
collects:

.. code-block:: rust

    use std::sync::mpsc;
    use std::thread;

    fn main() {
        let (tx, rx) = mpsc::channel();

        for i in 0..3 {
            let tx = tx.clone();
            thread::spawn(move || {
                tx.send(i).unwrap();
            });
        }
        drop(tx);  // drop original sender

        for msg in rx {
            println!("{}", msg);
        }
    }

Send and Sync Traits
--------------------

Rust uses marker traits to ensure thread safety at compile time. ``Send`` means a type
can be transferred to another thread (ownership can cross thread boundaries). ``Sync``
means a type can be shared between threads (multiple threads can have references to it).
Most types are automatically ``Send`` and ``Sync``, but types like ``Rc`` (non-atomic
reference counting) and ``RefCell`` (non-thread-safe interior mutability) are not:

.. code-block:: rust

    use std::rc::Rc;
    use std::sync::Arc;

    // Rc is NOT Send or Sync (not thread-safe)
    let rc = Rc::new(42);
    // thread::spawn(move || { println!("{}", rc); });  // compile error!

    // Arc IS Send and Sync
    let arc = Arc::new(42);
    thread::spawn(move || { println!("{}", arc); });  // OK

    // Raw pointers are NOT Send or Sync
    let ptr: *const i32 = &42;
    // thread::spawn(move || { unsafe { println!("{}", *ptr); } });  // error!

Scoped Threads
--------------

Scoped threads (``thread::scope``) can borrow from the parent stack because they're
guaranteed to complete before the scope exits. This eliminates the need for ``Arc``
when you just want to share read-only data with spawned threads. All threads spawned
within the scope are automatically joined when the scope ends, making it impossible
to forget to join:

.. code-block:: rust

    use std::thread;

    fn main() {
        let data = vec![1, 2, 3];

        thread::scope(|s| {
            s.spawn(|| {
                // Can borrow data without move
                println!("{:?}", data);
            });

            s.spawn(|| {
                println!("{:?}", data);
            });
        });
        // All spawned threads joined here

        println!("data still valid: {:?}", data);
    }

Thread Pool Pattern
-------------------

A thread pool maintains a set of worker threads that process jobs from a queue. This
avoids the overhead of creating new threads for each task. The following example shows
a basic implementation using channels - workers receive jobs through a channel and
execute them. Production code would typically use a crate like ``rayon`` or ``threadpool``:

.. code-block:: rust

    use std::sync::{mpsc, Arc, Mutex};
    use std::thread;

    type Job = Box<dyn FnOnce() + Send + 'static>;

    struct ThreadPool {
        workers: Vec<thread::JoinHandle<()>>,
        sender: mpsc::Sender<Job>,
    }

    impl ThreadPool {
        fn new(size: usize) -> Self {
            let (sender, receiver) = mpsc::channel();
            let receiver = Arc::new(Mutex::new(receiver));
            let mut workers = Vec::with_capacity(size);

            for _ in 0..size {
                let receiver = Arc::clone(&receiver);
                workers.push(thread::spawn(move || loop {
                    let job = receiver.lock().unwrap().recv();
                    match job {
                        Ok(job) => job(),
                        Err(_) => break,
                    }
                }));
            }

            ThreadPool { workers, sender }
        }

        fn execute<F>(&self, f: F)
        where
            F: FnOnce() + Send + 'static,
        {
            self.sender.send(Box::new(f)).unwrap();
        }
    }

Atomic Types
------------

Atomic types provide lock-free thread-safe operations on primitive values. They're
faster than mutexes for simple operations like counters because they use CPU atomic
instructions rather than OS-level locks. The ``Ordering`` parameter specifies memory
ordering guarantees - ``SeqCst`` (sequentially consistent) is the safest but slowest;
``Relaxed`` is fastest but provides minimal guarantees:

.. code-block:: rust

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    fn main() {
        let counter = AtomicUsize::new(0);

        thread::scope(|s| {
            for _ in 0..10 {
                s.spawn(|| {
                    counter.fetch_add(1, Ordering::SeqCst);
                });
            }
        });

        println!("{}", counter.load(Ordering::SeqCst));  // 10
    }

See Also
--------

- :doc:`rust_smartptr` - Arc and Mutex details
- :doc:`rust_closure` - Move closures for threads
