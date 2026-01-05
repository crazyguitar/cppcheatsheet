========
Iterator
========

.. contents:: Table of Contents
    :backlinks: none

Iterator Categories
-------------------

:Source: `src/iterator/basics <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/iterator/basics>`_

Iterators are the glue between containers and algorithms in C++. They provide
a uniform way to traverse and access elements regardless of the underlying
container type. This abstraction allows the same algorithm (like ``std::sort``
or ``std::find``) to work with vectors, lists, arrays, and even custom containers.

The standard library defines **five iterator categories**, each building upon the
capabilities of the previous one. Understanding these categories is essential
for writing generic code and choosing the right algorithm for your container.

.. list-table:: Iterator Categories Summary
   :header-rows: 1
   :widths: 20 45 35

   * - Category
     - Capabilities
     - Example Containers
   * - Input
     - Read-only, single-pass, forward only (``++``, ``*``, ``==``)
     - ``istream_iterator``
   * - Output
     - Write-only, single-pass, forward only (``++``, ``*=``)
     - ``ostream_iterator``, ``back_inserter``
   * - Forward
     - Read/write, multi-pass, forward only (``++``)
     - ``forward_list``, ``unordered_set``
   * - Bidirectional
     - Forward + backward (``++``, ``--``)
     - ``list``, ``set``, ``map``
   * - Random Access
     - Bidirectional + arithmetic (``+``, ``-``, ``[]``)
     - ``vector``, ``array``, ``deque``

.. note::

    Iterator categories form a hierarchy: Random Access > Bidirectional > Forward > Input/Output.
    An algorithm requiring a Forward iterator will work with Bidirectional or Random Access
    iterators, but not with Input or Output iterators.

**Input Iterator**

Input iterators support **read-only**, **single-pass**, **forward-only** traversal. They
can only be dereferenced to read values (not write), and once you advance past
an element, **you cannot go back to it**. This makes them suitable for reading from
streams where data is consumed as it's read. The classic example is
``std::istream_iterator``, which reads from an input stream:

.. code-block:: cpp

    #include <iostream>
    #include <iterator>
    #include <sstream>

    int main(int argc, char *argv[]) {
      std::istringstream iss("1 2 3");
      std::istream_iterator<int> it(iss), end;

      while (it != end) {
        std::cout << *it << " ";  // Read each value once
        ++it;
      }
      // Output: 1 2 3
    }

.. warning::

    Input iterators are **single-pass**: once you call ``++it``, the previous position
    is gone forever. Copying an input iterator and advancing one copy **may invalidate
    the other copy**.

**Output Iterator**

Output iterators are the **write-only** counterpart to input iterators. They support
single-pass, forward-only traversal, but can only be dereferenced to write values
(**not read**). You cannot read back what you wrote or revisit previous positions.
The typical example is ``std::ostream_iterator``, which writes to an output stream.
Output iterators are commonly used as destinations for algorithms like ``std::copy``:

.. code-block:: cpp

    #include <iostream>
    #include <iterator>
    #include <vector>

    int main(int argc, char *argv[]) {
      std::ostream_iterator<int> out(std::cout, " ");

      *out = 1; ++out;
      *out = 2; ++out;
      *out = 3; ++out;
      // Output: 1 2 3
    }

**Forward Iterator**

Forward iterators combine the capabilities of input and output iterators: they
support **both reading and writing**, and they allow **multi-pass traversal**. This means
you can iterate through a range multiple times, and the iterators remain valid.
However, they **only move forward** (no ``--`` operator). ``std::forward_list`` provides
forward iterators, as its singly-linked structure doesn't support backward traversal:

.. code-block:: cpp

    #include <forward_list>
    #include <iostream>

    int main(int argc, char *argv[]) {
      std::forward_list<int> fl{1, 2, 3};

      // Can traverse multiple times
      for (auto it = fl.begin(); it != fl.end(); ++it) {
        *it *= 2;  // Read and write
      }
      for (auto it = fl.begin(); it != fl.end(); ++it) {
        std::cout << *it << " ";  // 2 4 6
      }
    }

**Bidirectional Iterator**

Bidirectional iterators extend forward iterators by adding the ability to **move
backward** using the decrement operator (``--``). This enables algorithms that need
to traverse in both directions, such as ``std::reverse`` or backward searches.
``std::list`` and ``std::set`` provide bidirectional iterators. The doubly-linked
structure of ``std::list`` naturally supports movement in both directions:

.. code-block:: cpp

    #include <iostream>
    #include <list>

    int main(int argc, char *argv[]) {
      std::list<int> l{1, 2, 3, 4, 5};

      auto it = l.end();
      while (it != l.begin()) {
        --it;  // Can move backward
        std::cout << *it << " ";  // 5 4 3 2 1
      }
    }

**Random Access Iterator**

Random access iterators are the **most powerful category**, adding **constant-time
arithmetic operations** to bidirectional iterators. You can jump forward or backward
by any number of positions using ``+``, ``-``, ``+=``, ``-=``, and access elements
at arbitrary offsets using subscript ``[]``. You can also compute the **distance
between two iterators in O(1) time**. ``std::vector``, ``std::array``, and
``std::deque`` provide random access iterators. This is what enables efficient
algorithms like ``std::sort`` (which **requires random access** for O(n log n) performance):

.. code-block:: cpp

    #include <iostream>
    #include <vector>

    int main(int argc, char *argv[]) {
      std::vector<int> v{1, 2, 3, 4, 5};

      auto it = v.begin();
      it += 3;                    // Jump forward by 3
      std::cout << *it << "\n";   // 4

      it -= 2;                    // Jump backward by 2
      std::cout << *it << "\n";   // 2

      std::cout << it[2] << "\n"; // Subscript: 4
      std::cout << (v.end() - v.begin()) << "\n";  // Distance: 5
    }

.. note::

    ``std::sort`` **requires random access iterators**. You cannot use ``std::sort``
    on ``std::list``; use ``list.sort()`` member function instead.

Iterator Operations
-------------------

The ``<iterator>`` header provides utility functions that work across different
iterator categories. These functions abstract away the differences between
iterator types, allowing you to write **generic code** that works with any iterator:

- ``std::advance(it, n)``: Moves iterator ``it`` by ``n`` positions (**modifies in place**)
- ``std::distance(first, last)``: Returns the number of elements between two iterators
- ``std::next(it, n=1)``: Returns a **new iterator** ``n`` positions after ``it``
- ``std::prev(it, n=1)``: Returns a **new iterator** ``n`` positions before ``it``

These functions **automatically use the most efficient implementation** based on the
iterator category. For random access iterators, ``std::advance`` uses ``+=`` (**O(1)**),
while for other iterators it uses repeated ``++`` (**O(n)**).

.. code-block:: cpp

    #include <iostream>
    #include <iterator>
    #include <vector>

    int main(int argc, char *argv[]) {
      std::vector<int> v{1, 2, 3, 4, 5};

      // std::advance - move iterator by n positions
      auto it = v.begin();
      std::advance(it, 2);
      std::cout << *it << "\n";  // 3

      // std::distance - count elements between iterators
      auto dist = std::distance(v.begin(), it);
      std::cout << dist << "\n"; // 2

      // std::next / std::prev - return new iterator
      auto next_it = std::next(it);     // Points to 4
      auto prev_it = std::prev(it);     // Points to 2
      std::cout << *next_it << " " << *prev_it << "\n";  // 4 2
    }

.. tip::

    Prefer ``std::next(it)`` over ``it + 1`` in generic code. ``std::next`` works
    with **any forward iterator**, while ``+`` **only works with random access iterators**.

lower_bound and upper_bound
---------------------------

:Source: `src/iterator/bounds <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/iterator/bounds>`_

Binary search algorithms ``std::lower_bound`` and ``std::upper_bound`` are
essential for working with **sorted ranges**. They use binary search to find
elements in **O(log n)** time for random access iterators. Understanding the
subtle difference between them is crucial:

- ``lower_bound(first, last, val)``: Returns iterator to **first element >= val**
- ``upper_bound(first, last, val)``: Returns iterator to **first element > val**

Together they define the range of elements equal to ``val``: ``[lower_bound, upper_bound)``.
If the value doesn't exist, both return the **position where it would be inserted**
to maintain sorted order. This makes them ideal for insertion and deletion in
sorted containers.

.. code-block:: cpp

    #include <algorithm>
    #include <iostream>
    #include <vector>

    int main(int argc, char *argv[]) {
      std::vector<int> v{1, 2, 3, 4, 5, 7, 10};

      auto x = 5;
      auto lo = std::lower_bound(v.begin(), v.end(), x);
      auto hi = std::upper_bound(v.begin(), v.end(), x);

      std::cout << "lower_bound(5): " << *lo << "\n";  // 5 (>= 5)
      std::cout << "upper_bound(5): " << *hi << "\n";  // 7 (> 5)

      // Insert into sorted container
      auto pos = std::upper_bound(v.begin(), v.end(), 6);
      v.insert(pos, 6);  // v is now {1, 2, 3, 4, 5, 6, 7, 10}

      // Erase from sorted container
      pos = std::lower_bound(v.begin(), v.end(), 4);
      v.erase(pos);      // v is now {1, 2, 3, 5, 6, 7, 10}
    }

Iterator Invalidation
---------------------

:Source: `src/iterator/invalidation <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/iterator/invalidation>`_

Understanding when iterators become invalid is **crucial for avoiding undefined
behavior**. Different containers have different invalidation rules:

**std::vector:**

- ``push_back()``: Invalidates **all iterators if reallocation occurs**
- ``insert()``: Invalidates iterators **at and after** the insertion point
- ``erase()``: Invalidates iterators **at and after** the erased element

**std::deque:**

- Insertion/erasure at ends: Only invalidates iterators to the affected end
- Insertion/erasure in middle: **Invalidates all iterators**

**std::list:**

- **Only iterators to erased elements are invalidated**
- Other iterators remain valid after any operation

.. code-block:: cpp

    #include <iostream>
    #include <list>
    #include <vector>

    int main(int argc, char *argv[]) {
      // Vector: iterator invalidated after erase
      std::vector<int> v{1, 2, 3, 4, 5};
      auto vit = v.begin() + 2;
      v.erase(vit);
      // vit is now invalid! Use returned iterator instead:
      // vit = v.erase(vit);

      // List: only erased iterator is invalidated
      std::list<int> l{1, 2, 3, 4, 5};
      auto lit = l.begin();
      auto next = std::next(lit);
      l.erase(lit);
      // next is still valid
      std::cout << *next << "\n";  // 2
    }

.. warning::

    Using an invalidated iterator is **undefined behavior**. Always use the iterator
    returned by ``erase()`` to continue iteration: ``it = container.erase(it);``

.. tip::

    When iterating and erasing, use the **erase-remove idiom** or iterate carefully:

    .. code-block:: cpp

        // Safe pattern for erasing while iterating
        for (auto it = v.begin(); it != v.end(); ) {
          if (should_erase(*it)) {
            it = v.erase(it);  // erase returns next valid iterator
          } else {
            ++it;
          }
        }

Custom Iterator
---------------

:Source: `src/iterator/custom <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/iterator/custom>`_

Creating a custom iterator requires implementing the iterator interface. For
a forward iterator, you need: ``operator*``, ``operator->``, ``operator++``,
``operator==``, and ``operator!=``. C++20 simplifies this with default comparisons.

.. note::

    Custom iterators must define five type aliases (``iterator_category``,
    ``value_type``, ``difference_type``, ``pointer``, ``reference``) to work
    with standard algorithms and ``std::iterator_traits``.

.. code-block:: cpp

    #include <iostream>
    #include <iterator>
    #include <memory>

    template <typename T>
    class Array {
     public:
      class iterator {
       public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T *;
        using reference = T &;

        iterator(T *ptr) : ptr_{ptr} {}
        reference operator*() { return *ptr_; }
        pointer operator->() { return ptr_; }
        iterator &operator++() { ++ptr_; return *this; }
        iterator operator++(int) { auto tmp = *this; ++ptr_; return tmp; }
        bool operator==(const iterator &rhs) const { return ptr_ == rhs.ptr_; }
        bool operator!=(const iterator &rhs) const { return ptr_ != rhs.ptr_; }

       private:
        T *ptr_;
      };

      Array(size_t size) : size_(size), data_{std::make_unique<T[]>(size)} {}
      size_t size() const { return size_; }
      T &operator[](size_t i) { return data_[i]; }
      iterator begin() { return iterator(data_.get()); }
      iterator end() { return iterator(data_.get() + size_); }

     private:
      size_t size_;
      std::unique_ptr<T[]> data_;
    };

    int main(int argc, char *argv[]) {
      Array<int> arr(3);
      arr[0] = 10;
      arr[1] = 20;
      arr[2] = 30;

      for (auto &x : arr) {
        std::cout << x << " ";
      }
      std::cout << "\n";
    }

C++20 Ranges Overview
---------------------

C++20 introduced the Ranges library (``<ranges>``) which provides a modern,
composable way to work with sequences. Ranges are an abstraction over
iterator pairs, and views are lazy, non-owning ranges that can be composed
using the pipe operator (``|``).

**Key concepts:**

- **Range**: Anything with ``begin()`` and ``end()`` (containers, arrays, views)
- **View**: A lightweight, non-owning range with O(1) copy/move
- **Range Adaptor**: A function that takes a range and returns a view
- **Pipe Operator**: Composes adaptors: ``range | adaptor1 | adaptor2``

.. note::

    Views are **lazy**: no computation happens until you iterate. This means you can
    chain multiple views **without creating intermediate containers**, and only the
    elements you actually access are processed.

.. warning::

    Views **do not own their data**. The underlying range must outlive the view.
    Returning a view to a local container is **undefined behavior**.

views::iota
-----------

:Source: `src/iterator/iota <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/iterator/iota>`_

``std::views::iota`` generates a sequence of **incrementing values**. It can be
bounded or **unbounded (infinite)**. This is useful for generating index sequences
or replacing traditional for loops.

.. code-block:: cpp

    #include <iostream>
    #include <ranges>

    int main(int argc, char *argv[]) {
      // Bounded: [1, 6)
      for (auto i : std::views::iota(1, 6)) {
        std::cout << i << " ";  // 1 2 3 4 5
      }
      std::cout << "\n";

      // Unbounded with take
      for (auto i : std::views::iota(1) | std::views::take(5)) {
        std::cout << i << " ";  // 1 2 3 4 5
      }
      std::cout << "\n";
    }

.. tip::

    Use ``std::views::iota(0, n)`` instead of ``for (int i = 0; i < n; ++i)``
    when you need a range-based approach or want to **compose with other views**.

views::filter
-------------

:Source: `src/iterator/filter <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/iterator/filter>`_

``std::views::filter`` creates a view containing only elements that satisfy
a predicate. The predicate is evaluated lazily during iteration.

.. code-block:: cpp

    #include <iostream>
    #include <ranges>
    #include <vector>

    int main(int argc, char *argv[]) {
      std::vector<int> v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

      auto evens = v | std::views::filter([](int x) { return x % 2 == 0; });

      for (auto x : evens) {
        std::cout << x << " ";  // 2 4 6 8 10
      }
      std::cout << "\n";
    }

views::transform
----------------

:Source: `src/iterator/transform <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/iterator/transform>`_

``std::views::transform`` applies a function to each element, creating a view
of the transformed values. Similar to ``map`` in functional programming.

.. code-block:: cpp

    #include <iostream>
    #include <ranges>
    #include <vector>

    int main(int argc, char *argv[]) {
      std::vector<int> v{1, 2, 3, 4, 5};

      auto squares = v | std::views::transform([](int x) { return x * x; });

      for (auto x : squares) {
        std::cout << x << " ";  // 1 4 9 16 25
      }
      std::cout << "\n";
    }

views::take and views::drop
---------------------------

:Source: `src/iterator/take-drop <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/iterator/take-drop>`_

``std::views::take(n)`` returns the first n elements, while ``std::views::drop(n)``
skips the first n elements. Both are useful for slicing ranges.

.. code-block:: cpp

    #include <iostream>
    #include <ranges>
    #include <vector>

    int main(int argc, char *argv[]) {
      std::vector<int> v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

      // First 3 elements
      for (auto x : v | std::views::take(3)) {
        std::cout << x << " ";  // 1 2 3
      }
      std::cout << "\n";

      // Skip first 3 elements
      for (auto x : v | std::views::drop(3)) {
        std::cout << x << " ";  // 4 5 6 7 8 9 10
      }
      std::cout << "\n";

      // Slice [3, 6): drop 3, take 3
      for (auto x : v | std::views::drop(3) | std::views::take(3)) {
        std::cout << x << " ";  // 4 5 6
      }
      std::cout << "\n";
    }

views::reverse
--------------

:Source: `src/iterator/reverse <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/iterator/reverse>`_

``std::views::reverse`` creates a view that iterates in reverse order. It
requires a bidirectional range.

.. code-block:: cpp

    #include <iostream>
    #include <ranges>
    #include <string>
    #include <vector>

    int main(int argc, char *argv[]) {
      std::vector<int> v{1, 2, 3, 4, 5};

      for (auto x : v | std::views::reverse) {
        std::cout << x << " ";  // 5 4 3 2 1
      }
      std::cout << "\n";

      // Reverse a string
      std::string s = "hello";
      for (auto c : s | std::views::reverse) {
        std::cout << c;  // olleh
      }
      std::cout << "\n";
    }

views::split and views::join
----------------------------

:Source: `src/iterator/split-join <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/iterator/split-join>`_

``std::views::split`` divides a range by a delimiter, while ``std::views::join``
flattens a range of ranges into a single range.

.. code-block:: cpp

    #include <iostream>
    #include <ranges>
    #include <string>
    #include <string_view>
    #include <vector>

    int main(int argc, char *argv[]) {
      // Split string by delimiter
      std::string s = "one,two,three";
      for (auto word : s | std::views::split(',')) {
        std::cout << std::string_view(word) << "\n";
      }

      // Join nested ranges
      std::vector<std::string> words = {"hello", "world"};
      for (auto c : words | std::views::join) {
        std::cout << c;  // helloworld
      }
      std::cout << "\n";
    }

views::enumerate (C++23)
------------------------

:Source: `src/iterator/enumerate <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/iterator/enumerate>`_

``std::views::enumerate`` pairs each element with its index. This is available
in C++23. For C++20, you can use ``std::views::iota`` with ``std::views::zip``
or a simple counter.

.. code-block:: cpp

    #include <iostream>
    #include <ranges>
    #include <vector>

    int main(int argc, char *argv[]) {
      std::vector<std::string> v{"apple", "banana", "cherry"};

    #if __cplusplus >= 202302L
      // C++23: views::enumerate
      for (auto [i, val] : v | std::views::enumerate) {
        std::cout << i << ": " << val << "\n";
      }
    #else
      // C++20 workaround
      for (size_t i = 0; auto &val : v) {
        std::cout << i++ << ": " << val << "\n";
      }
    #endif
    }

views::zip (C++23)
------------------

:Source: `src/iterator/zip <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/iterator/zip>`_

``std::views::zip`` combines multiple ranges into a range of tuples. Available
in C++23.

.. code-block:: cpp

    #include <iostream>
    #include <ranges>
    #include <vector>

    int main(int argc, char *argv[]) {
      std::vector<int> a{1, 2, 3};
      std::vector<std::string> b{"one", "two", "three"};

    #if __cplusplus >= 202302L
      // C++23: views::zip
      for (auto [x, y] : std::views::zip(a, b)) {
        std::cout << x << " -> " << y << "\n";
      }
    #else
      // C++20 workaround
      for (size_t i = 0; i < a.size(); ++i) {
        std::cout << a[i] << " -> " << b[i] << "\n";
      }
    #endif
    }

Composing Views
---------------

:Source: `src/iterator/compose <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/iterator/compose>`_

The power of ranges comes from composing multiple views. Views are lazy, so
no intermediate containers are created. The pipeline is evaluated element by
element during iteration.

.. code-block:: cpp

    #include <iostream>
    #include <ranges>
    #include <vector>

    int main(int argc, char *argv[]) {
      std::vector<int> v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

      // Filter evens, square them, take first 3
      auto result = v
          | std::views::filter([](int x) { return x % 2 == 0; })
          | std::views::transform([](int x) { return x * x; })
          | std::views::take(3);

      for (auto x : result) {
        std::cout << x << " ";  // 4 16 36
      }
      std::cout << "\n";
    }

Converting Views to Containers
------------------------------

:Source: `src/iterator/to-container <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/iterator/to-container>`_

Views are lazy and non-owning. To materialize results into a container, use
``std::ranges::to`` (C++23) or manual construction.

.. code-block:: cpp

    #include <iostream>
    #include <ranges>
    #include <vector>

    int main(int argc, char *argv[]) {
      auto view = std::views::iota(1, 6)
          | std::views::transform([](int x) { return x * x; });

    #if __cplusplus >= 202302L
      // C++23: ranges::to
      auto v = view | std::ranges::to<std::vector>();
    #else
      // C++20 workaround
      std::vector<int> v(view.begin(), view.end());
    #endif

      for (auto x : v) {
        std::cout << x << " ";  // 1 4 9 16 25
      }
      std::cout << "\n";
    }

Range Algorithms
----------------

:Source: `src/iterator/algorithms <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/iterator/algorithms>`_

C++20 also provides range-based versions of standard algorithms in the
``std::ranges`` namespace. These accept ranges directly instead of iterator pairs.

.. code-block:: cpp

    #include <algorithm>
    #include <iostream>
    #include <ranges>
    #include <vector>

    int main(int argc, char *argv[]) {
      std::vector<int> v{3, 1, 4, 1, 5, 9, 2, 6};

      // Range-based sort
      std::ranges::sort(v);

      // Range-based find
      auto it = std::ranges::find(v, 5);
      if (it != v.end()) {
        std::cout << "Found: " << *it << "\n";
      }

      // Range-based all_of
      bool all_positive = std::ranges::all_of(v, [](int x) { return x > 0; });
      std::cout << "All positive: " << all_positive << "\n";

      // Range-based count_if
      auto count = std::ranges::count_if(v, [](int x) { return x % 2 == 0; });
      std::cout << "Even count: " << count << "\n";
    }
