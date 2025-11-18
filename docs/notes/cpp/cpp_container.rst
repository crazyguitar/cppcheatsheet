=========
Container
=========

Priority Queue
--------------

.. contents:: Table of Contents
    :backlinks: none

Priority Queue
--------------

.. code-block:: cpp

    #include <iostream>
    #include <functional>
    #include <vector>
    #include <queue>

    template<typename Q>
    void dump(Q &q) {
      while(!q.empty()) {
        std::cout << q.top() << " ";
        q.pop();
      }
      std::cout << "\n";
    }

    void foo() {
      std::vector<int> data{1, 5, 2, 1, 3};
      std::priority_queue<int> queue;
      for (auto & x : data) { queue.push(x); }
      dump(queue);
    }

    void bar() {
      std::vector<int> data{1, 5, 2, 1, 3};
      std::priority_queue<int, std::vector<int>, std::greater<int>> queue;
      for (auto & x : data) { queue.push(x); }
      dump(queue);
    }

    void baz() {
      std::vector<int> data{1, 5, 2, 1, 3};
      auto cmp = [](int x, int y) { return x < y; };
      std::priority_queue<int, std::vector<int>, decltype(cmp)> queue(cmp);
      for (auto & x : data) { queue.push(x); }
      dump(queue);
    }

    int main(int argc, char *argv[]) {
      foo();
      bar();
      baz();
      // 5 3 2 1 1
      // 1 1 2 3 5
      // 1 1 2 3 5
    }


Priority queue is useful when a programmer need to merge multiple lists of data
in order.

.. code-block:: cpp

    #include <iostream>
    #include <vector>
    #include <queue>

    template<typename Q>
    void dump(Q &q) {
      while(!q.empty()) {
        std::cout << q.top() << " ";
        q.pop();
      }
      std::cout << "\n";
    }

    int main(int argc, char *argv[]) {
      std::priority_queue<int> queue;
      std::vector<int> x{9, 7, 8};
      std::vector<int> y{0, 5, 3};
      for (auto &e : x) { queue.push(e); }
      for (auto &e : y) { queue.push(e); }
      dump(queue);
      // 9 8 7 5 3 0
    }

Profiling
---------

.. code-block:: cpp

    // profile.h
    #include <iostream>
    #include <chrono>

    using milliseconds = std::chrono::milliseconds;

    template <typename T, typename F>
    void profile(T &t, F &func) {
      const auto start = std::chrono::steady_clock::now();
      func(t);
      const auto end = std::chrono::steady_clock::now();
      const auto d = end - start;
      const auto mill = std::chrono::duration_cast<milliseconds>(d).count();
      std::cout << mill << " ms\n";
    }

Push Front
``````````

.. code-block:: cpp

    // g++ -O3 -std=c++17 -Wall -Werror -I${HDR} a.cpp

    #include <vector>
    #include <deque>
    #include <list>
    #include <range/v3/view/iota.hpp>
    #include "profile.h"

    template <typename T>
    void insert(T &t) {
      for (auto i : ranges::views::iota(0, 300000)) {
        t.insert(t.begin(), i);
      }
    }

    int main(int argc, char *argv[]) {
      std::vector<int> v;
      std::deque<int> q;
      std::list<int> l;
      profile(v, insert<decltype(v)>);
      profile(q, insert<decltype(q)>);
      profile(l, insert<decltype(l)>);
    }

.. code-block:: bash

    $ ./a.out
    16045 ms
    1 ms
    6 ms


Push Back
`````````

.. code-block:: cpp

    #include <vector>
    #include <deque>
    #include <list>
    #include <range/v3/view/iota.hpp>
    #include "profile.h"

    template <typename T>
    void insert(T &t) {
      for (auto i : ranges::views::iota(0, 1000000)) {
        t.push_back(i);
      }
    }

    int main(int argc, char *argv[]) {
      std::vector<int> v;
      std::deque<int> q;
      std::list<int> l;
      profile(v, insert<decltype(v)>);
      profile(q, insert<decltype(q)>);
      profile(l, insert<decltype(l)>);
    }

.. code-block:: bash

    ./a.out
    7 ms
    2 ms
    39 ms

Pop Front
`````````

.. code-block:: cpp

    #include <vector>
    #include <deque>
    #include <list>
    #include <range/v3/view/iota.hpp>
    #include "profile.h"

    template <typename T>
    void insert(T &t) {
      for (auto i : ranges::views::iota(0, 300000)) {
        t.push_back(i);
      }
    }

    template <typename T>
    void pop_front(T &t) {
      while (!t.empty()) {
        t.pop_front();
      }
    }

    template <typename T>
    void erase(T &v) {
      while(!v.empty()) {
        v.erase(v.begin());
      }
    }

    int main(int argc, char *argv[]) {
      std::vector<int> v;
      std::deque<int> q;
      std::list<int> l;
      insert(v); insert(q); insert(l);
      profile(v, erase<decltype(v)>);
      profile(q, pop_front<decltype(q)>);
      profile(l, pop_front<decltype(l)>);
    }


.. code-block:: bash

    $ ./a.out
    22923 ms
    0 ms
    12 ms

Pop Back
````````

.. code-block:: cpp

    #include <vector>
    #include <deque>
    #include <list>
    #include <range/v3/view/iota.hpp>
    #include "profile.h"

    template <typename T>
    void insert(T &t) {
      for (auto i : ranges::views::iota(0, 1000000)) {
        t.push_back(i);
      }
    }

    template <typename T>
    void pop_back(T &t) {
      while (!t.empty()) {
        t.pop_back();
      }
    }

    int main(int argc, char *argv[]) {
      std::vector<int> v;
      std::deque<int> q;
      std::list<int> l;
      insert(v); insert(q); insert(l);
      profile(v, pop_back<decltype(v)>);
      profile(q, pop_back<decltype(q)>);
      profile(l, pop_back<decltype(l)>);
    }

.. code-block:: bash

    $ ./a.out
    0 ms
    0 ms
    30 ms
