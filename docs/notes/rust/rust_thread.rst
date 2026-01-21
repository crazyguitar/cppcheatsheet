=======
Threads
=======

.. meta::
   :description: Rust threading compared to C++. Covers std::thread, Arc, Mutex, channels, and Send/Sync traits.
   :keywords: Rust, threads, concurrency, Arc, Mutex, channels, Send, Sync, C++ threading

.. contents:: Table of Contents
    :backlinks: none

:Source: `src/rust/threads <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/threads>`_

Rust's ownership system prevents data races at compile time. The ``Send`` and
``Sync`` traits ensure thread-safe data sharing.

Basic Threading
---------------

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

Rust uses marker traits to ensure thread safety:

- ``Send``: Type can be transferred to another thread
- ``Sync``: Type can be shared between threads (``&T`` is ``Send``)

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

Scoped threads can borrow from the parent stack:

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
