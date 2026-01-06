=========
Algorithm
=========

.. meta::
   :description: C++ STL algorithm reference with examples for sorting, searching, transforming, and manipulating containers.
   :keywords: C++, STL, algorithm, sort, find, transform, copy, remove, unique, binary_search

.. contents:: Table of Contents
    :backlinks: none

Introduction
------------

The ``<algorithm>`` header provides a collection of functions for searching, sorting,
counting, and manipulating ranges of elements. These algorithms operate on iterator
pairs and are designed to work with any container that provides compatible iterators.
This separation of algorithms from containers is a fundamental design principle of
the C++ Standard Template Library (STL), enabling code reuse and generic programming.

C++20 introduced the Ranges library which provides a more composable and safer
alternative to iterator-based algorithms. For comprehensive coverage of C++20 ranges
and views, see :doc:`cpp_iterator`.

Complexity Summary
------------------

The following table summarizes the time complexity and iterator requirements for
common STL algorithms. Understanding these characteristics helps you choose the
right algorithm for your use case and predict performance at scale.

The **Iterator Required** column indicates the minimum iterator category needed.
Algorithms work with that category or higher (Random Access > Bidirectional >
Forward > Input). See the `Iterator Categories <cpp_iterator.html#iterator-categories>`_
section for details on each category and their capabilities.

.. list-table::
   :header-rows: 1
   :widths: 25 20 25 30

   * - Algorithm
     - Time Complexity
     - Iterator Required
     - Notes
   * - ``sort``
     - O(n log n)
     - Random Access
     - Not stable; use ``stable_sort`` if order matters
   * - ``stable_sort``
     - O(n log n)
     - Random Access
     - O(n log² n) if no extra memory available
   * - ``partial_sort``
     - O(n log k)
     - Random Access
     - k = number of elements to sort
   * - ``nth_element``
     - O(n) average
     - Random Access
     - O(n²) worst case
   * - ``find``
     - O(n)
     - Input
     - Use ``binary_search`` for sorted ranges
   * - ``find_if``
     - O(n)
     - Input
     - Short-circuits on first match
   * - ``binary_search``
     - O(log n)
     - Forward
     - Requires sorted range
   * - ``lower_bound``
     - O(log n)
     - Forward
     - O(n) for non-random access iterators
   * - ``upper_bound``
     - O(log n)
     - Forward
     - O(n) for non-random access iterators
   * - ``count``
     - O(n)
     - Input
     - Full range traversal
   * - ``count_if``
     - O(n)
     - Input
     - Full range traversal
   * - ``transform``
     - O(n)
     - Input/Output
     - One function call per element
   * - ``for_each``
     - O(n)
     - Input
     - One function call per element
   * - ``accumulate``
     - O(n)
     - Input
     - Sequential left-to-right
   * - ``reduce``
     - O(n)
     - Input
     - Parallelizable (C++17)
   * - ``remove``
     - O(n)
     - Forward
     - Does not resize container
   * - ``remove_if``
     - O(n)
     - Forward
     - Does not resize container
   * - ``unique``
     - O(n)
     - Forward
     - Removes consecutive duplicates only
   * - ``copy``
     - O(n)
     - Input/Output
     - Destination must have space
   * - ``copy_if``
     - O(n)
     - Input/Output
     - One predicate call per element
   * - ``generate``
     - O(n)
     - Forward
     - One generator call per element
   * - ``iota``
     - O(n)
     - Forward
     - Sequential increment
   * - ``reverse``
     - O(n)
     - Bidirectional
     - In-place swap
   * - ``rotate``
     - O(n)
     - Forward
     - At most n swaps
   * - ``min_element``
     - O(n)
     - Forward
     - n-1 comparisons
   * - ``max_element``
     - O(n)
     - Forward
     - n-1 comparisons
   * - ``minmax_element``
     - O(n)
     - Forward
     - ~1.5n comparisons (more efficient)
   * - ``all_of``
     - O(n)
     - Input
     - Short-circuits on false
   * - ``any_of``
     - O(n)
     - Input
     - Short-circuits on true
   * - ``none_of``
     - O(n)
     - Input
     - Short-circuits on true
   * - ``set_union``
     - O(n + m)
     - Input
     - Requires sorted ranges
   * - ``set_intersection``
     - O(n + m)
     - Input
     - Requires sorted ranges
   * - ``set_difference``
     - O(n + m)
     - Input
     - Requires sorted ranges

Sorting
-------

``std::sort``
~~~~~~~~~~~~~

:Source: `src/algorithm/sort <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/algorithm/sort>`_

``std::sort`` rearranges elements in ascending order by default using an introsort
algorithm (a hybrid of quicksort, heapsort, and insertion sort) that guarantees
O(n log n) worst-case complexity. The algorithm requires random access iterators,
so it works with ``std::vector``, ``std::array``, and ``std::deque``, but not with
``std::list`` (use ``list.sort()`` instead). For custom ordering, provide a comparison
function or use ``std::greater<>`` for descending order. Note that ``std::sort`` is
**not stable**—equal elements may be reordered relative to their original positions.

.. code-block:: cpp

    #include <algorithm>
    #include <vector>

    int main() {
      std::vector v{3, 1, 4, 1, 5, 9, 2, 6};

      std::sort(v.begin(), v.end());  // ascending
      std::sort(v.begin(), v.end(), std::greater<>{});  // descending

      // custom comparator
      std::sort(v.begin(), v.end(), [](int a, int b) { return a > b; });
    }

``std::stable_sort``
~~~~~~~~~~~~~~~~~~~~

:Source: `src/algorithm/stable-sort <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/algorithm/stable-sort>`_

``std::stable_sort`` preserves the relative order of elements that compare equal,
which is essential when sorting by multiple criteria sequentially. For example,
if you first sort employees by name and then by department, employees within the
same department will remain sorted by name. The algorithm uses merge sort and
guarantees O(n log n) complexity when sufficient memory is available, or
O(n log² n) otherwise. Use ``std::stable_sort`` when the original ordering of
equivalent elements carries semantic meaning.

.. code-block:: cpp

    #include <algorithm>
    #include <vector>

    struct Item {
      int priority;
      int order;
    };

    int main() {
      std::vector<Item> items{{1, 1}, {2, 2}, {1, 3}, {2, 4}};

      // stable_sort preserves original order for equal elements
      std::stable_sort(items.begin(), items.end(),
                       [](auto &a, auto &b) { return a.priority < b.priority; });
      // items with same priority keep their original relative order
    }

``std::partial_sort``
~~~~~~~~~~~~~~~~~~~~~

:Source: `src/algorithm/partial-sort <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/algorithm/partial-sort>`_

``std::partial_sort`` sorts only the first N elements of a range, leaving the
remaining elements in an unspecified order. This is significantly more efficient
than a full sort when you only need the top N results—it runs in O(n log k) time
where k is the number of elements to sort, compared to O(n log n) for a full sort.
Common use cases include finding the top 10 scores, the 5 smallest values, or
implementing pagination where only the first page needs to be sorted.

.. code-block:: cpp

    #include <algorithm>
    #include <vector>

    int main() {
      std::vector v{5, 7, 4, 2, 8, 6, 1, 9, 0, 3};

      // sort only first 3 elements
      std::partial_sort(v.begin(), v.begin() + 3, v.end());
      // v: {0, 1, 2, ...} first 3 are sorted, rest unspecified
    }

``std::nth_element``
~~~~~~~~~~~~~~~~~~~~

:Source: `src/algorithm/nth-element <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/algorithm/nth-element>`_

``std::nth_element`` is a selection algorithm that partially sorts a range so that
the element at the nth position is exactly the element that would be there if the
range were fully sorted. All elements before the nth are less than or equal to it,
and all elements after are greater than or equal. This runs in O(n) average time,
making it ideal for finding medians, percentiles, or partitioning data around a
pivot value without the overhead of a full sort.

.. code-block:: cpp

    #include <algorithm>
    #include <vector>

    int main() {
      std::vector v{5, 6, 4, 3, 2, 6, 7, 9, 3};

      // find median (middle element)
      auto mid = v.begin() + v.size() / 2;
      std::nth_element(v.begin(), mid, v.end());
      // *mid is now the median value
    }

Searching
---------

``std::find``
~~~~~~~~~~~~~

:Source: `src/algorithm/find <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/algorithm/find>`_

``std::find`` performs a linear search through a range, returning an iterator to
the first element that equals the specified value, or ``end()`` if not found. The
algorithm has O(n) complexity and works with any input iterator, making it
universally applicable but potentially slow for large unsorted containers. For
sorted containers, prefer ``std::binary_search``, ``std::lower_bound``, or
``std::upper_bound`` which offer O(log n) complexity. For associative containers
like ``std::set`` or ``std::map``, use their member ``find()`` functions which
are optimized for their internal structure.

.. code-block:: cpp

    #include <algorithm>
    #include <iostream>
    #include <vector>

    int main() {
      std::vector v{1, 2, 3, 4, 5};

      auto it = std::find(v.begin(), v.end(), 3);
      if (it != v.end()) {
        std::cout << "Found: " << *it << "\n";
      }
    }

``std::find_if`` and ``std::find_if_not``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Source: `src/algorithm/find-if <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/algorithm/find-if>`_

``std::find_if`` searches for the first element satisfying a predicate function,
while ``std::find_if_not`` finds the first element that does **not** satisfy the
predicate. These are more flexible than ``std::find`` because they allow arbitrary
search conditions rather than simple equality. Common use cases include finding
the first negative number, the first element exceeding a threshold, or the first
invalid entry in a dataset. Both algorithms have O(n) complexity.

.. code-block:: cpp

    #include <algorithm>
    #include <iostream>
    #include <vector>

    int main() {
      std::vector v{1, 2, 3, 4, 5};

      auto even = std::find_if(v.begin(), v.end(), [](int x) { return x % 2 == 0; });
      auto odd = std::find_if_not(v.begin(), v.end(), [](int x) { return x % 2 == 0; });

      std::cout << "First even: " << *even << "\n";  // 2
      std::cout << "First odd: " << *odd << "\n";    // 1
    }

``std::binary_search``
~~~~~~~~~~~~~~~~~~~~~~

:Source: `src/algorithm/binary-search <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/algorithm/binary-search>`_

``std::binary_search`` checks whether a value exists in a **sorted** range using
binary search with O(log n) complexity. It returns a boolean indicating presence
but does not provide the element's position. To get an iterator to the element,
use ``std::lower_bound`` (first element >= value) or ``std::upper_bound`` (first
element > value). The ``std::equal_range`` function returns both bounds as a pair,
defining the range of elements equal to the search value. These functions are
essential for efficient searching and insertion in sorted containers.

.. code-block:: cpp

    #include <algorithm>
    #include <vector>

    int main() {
      std::vector v{1, 2, 3, 4, 5};

      bool found = std::binary_search(v.begin(), v.end(), 3);

      // lower_bound: first element >= value
      auto lb = std::lower_bound(v.begin(), v.end(), 3);

      // upper_bound: first element > value
      auto ub = std::upper_bound(v.begin(), v.end(), 3);

      // equal_range: pair of lower_bound and upper_bound
      auto [lo, hi] = std::equal_range(v.begin(), v.end(), 3);
    }

Counting
--------

``std::count`` and ``std::count_if``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Source: `src/algorithm/count <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/algorithm/count>`_

``std::count`` returns the number of elements in a range that equal a specified
value, while ``std::count_if`` counts elements satisfying a predicate. Both
algorithms traverse the entire range with O(n) complexity. These are useful for
frequency analysis, validation (checking how many elements meet criteria), and
statistical computations. For associative containers, the member ``count()``
function is more efficient as it leverages the container's internal structure.

.. code-block:: cpp

    #include <algorithm>
    #include <iostream>
    #include <vector>

    int main() {
      std::vector v{1, 2, 2, 3, 2, 4, 2};

      auto n = std::count(v.begin(), v.end(), 2);
      std::cout << "Count of 2: " << n << "\n";  // 4

      auto evens = std::count_if(v.begin(), v.end(), [](int x) { return x % 2 == 0; });
      std::cout << "Even count: " << evens << "\n";  // 5
    }

Transforming
------------

``std::transform``
~~~~~~~~~~~~~~~~~~

:Source: `src/algorithm/transform <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/algorithm/transform>`_

``std::transform`` applies a unary or binary function to elements and stores the
results in a destination range. It can transform elements in place (when source
and destination are the same) or into a separate container. This is the C++
equivalent of functional programming's ``map`` operation. Common uses include
converting strings to uppercase, scaling numeric values, extracting fields from
structures, or combining two ranges element-wise. The algorithm does not resize
the destination container, so ensure it has sufficient capacity.

.. code-block:: cpp

    #include <algorithm>
    #include <iostream>
    #include <string>
    #include <vector>

    int main() {
      std::string s = "Hello World";

      // transform in place
      std::transform(s.begin(), s.end(), s.begin(), ::toupper);
      std::cout << s << "\n";  // HELLO WORLD

      // transform to another container
      std::vector<int> v{1, 2, 3};
      std::vector<int> squared(v.size());
      std::transform(v.begin(), v.end(), squared.begin(), [](int x) { return x * x; });
    }

``std::for_each``
~~~~~~~~~~~~~~~~~

:Source: `src/algorithm/for-each <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/algorithm/for-each>`_

``std::for_each`` applies a function to each element in a range, primarily for
side effects like printing, accumulating values, or modifying elements in place.
Unlike ``std::transform``, it doesn't write to a destination range. The function
object is passed by value and returned after iteration, which can be useful for
stateful functors. In modern C++, range-based for loops often replace
``std::for_each`` for simple iteration, but the algorithm remains useful when
you need to pass a callable to another function or work with iterator pairs.

.. code-block:: cpp

    #include <algorithm>
    #include <iostream>
    #include <vector>

    int main() {
      std::vector v{1, 2, 3};

      std::for_each(v.begin(), v.end(), [](int &x) { x *= 2; });
      std::for_each(v.begin(), v.end(), [](int x) { std::cout << x << " "; });
      // Output: 2 4 6
    }


Accumulating
------------

``std::accumulate``
~~~~~~~~~~~~~~~~~~~

:Source: `src/algorithm/accumulate <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/algorithm/accumulate>`_

``std::accumulate`` (from ``<numeric>``) computes a single value from a range by
repeatedly applying a binary operation. By default, it sums elements starting from
an initial value, but custom operations enable products, string concatenation, or
any associative reduction. The initial value's type determines the result type, so
use ``0.0`` instead of ``0`` when accumulating floating-point values to avoid
truncation. Note that ``std::accumulate`` processes elements strictly left-to-right,
which matters for non-commutative operations.

.. code-block:: cpp

    #include <iostream>
    #include <numeric>
    #include <vector>

    int main() {
      std::vector v{1, 2, 3, 4, 5};

      int sum = std::accumulate(v.begin(), v.end(), 0);
      std::cout << "Sum: " << sum << "\n";  // 15

      int product = std::accumulate(v.begin(), v.end(), 1, std::multiplies<>{});
      std::cout << "Product: " << product << "\n";  // 120
    }

``std::reduce`` (C++17)
~~~~~~~~~~~~~~~~~~~~~~~

:Source: `src/algorithm/reduce <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/algorithm/reduce>`_

``std::reduce`` (from ``<numeric>``) is similar to ``std::accumulate`` but permits
out-of-order execution, enabling parallel implementations. The operation must be
both associative and commutative for correct results with parallel execution
policies. When used with ``std::execution::par``, it can significantly speed up
reductions on large datasets by utilizing multiple CPU cores. For sequential
execution, it behaves identically to ``std::accumulate``.

.. code-block:: cpp

    #include <iostream>
    #include <numeric>
    #include <vector>

    int main() {
      std::vector v{1, 2, 3, 4, 5};

      // reduce allows out-of-order execution (parallelizable)
      int sum = std::reduce(v.begin(), v.end(), 0);
      std::cout << "Sum: " << sum << "\n";
    }

Removing Elements
-----------------

Erase-Remove Idiom
~~~~~~~~~~~~~~~~~~

:Source: `src/algorithm/erase-remove <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/algorithm/erase-remove>`_

The erase-remove idiom is the classic way to remove elements from sequence
containers. ``std::remove`` and ``std::remove_if`` don't actually erase elements—
they move unwanted elements to the end of the range and return an iterator to the
new logical end. You must then call the container's ``erase()`` method to actually
remove them. This two-step process exists because algorithms work with iterators
and don't know about container memory management. The idiom is efficient, performing
the removal in O(n) time with a single pass through the data.

.. code-block:: cpp

    #include <algorithm>
    #include <vector>

    int main() {
      std::vector v{1, 2, 3, 2, 4, 2, 5};

      // erase-remove idiom
      v.erase(std::remove(v.begin(), v.end(), 2), v.end());
      // v: {1, 3, 4, 5}

      // with predicate
      std::vector v2{1, 2, 3, 4, 5, 6};
      v2.erase(std::remove_if(v2.begin(), v2.end(), [](int x) { return x % 2 == 0; }),
               v2.end());
      // v2: {1, 3, 5}
    }

``std::erase`` and ``std::erase_if`` (C++20)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Source: `src/algorithm/erase-cpp20 <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/algorithm/erase-cpp20>`_

C++20 introduced ``std::erase`` and ``std::erase_if`` as free functions that
combine the erase-remove idiom into a single call. These functions are overloaded
for each standard container type and handle the container-specific removal logic
automatically. They return the number of elements removed, which is useful for
logging or validation. This is now the preferred way to remove elements in
modern C++ code due to its clarity and reduced chance of errors.

.. code-block:: cpp

    #include <vector>

    int main() {
      std::vector v{1, 2, 3, 2, 4, 2, 5};

      // C++20: direct erase
      std::erase(v, 2);
      // v: {1, 3, 4, 5}

      std::vector v2{1, 2, 3, 4, 5, 6};
      std::erase_if(v2, [](int x) { return x % 2 == 0; });
      // v2: {1, 3, 5}
    }

Erasing from Associative Containers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Source: `src/algorithm/erase-map <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/algorithm/erase-map>`_

Associative containers (``std::map``, ``std::set``, etc.) require careful iteration
when erasing elements because erasing invalidates the iterator to the erased element.
The safe pattern is to use the iterator returned by ``erase()``, which points to
the next element. In C++20, ``std::erase_if`` works with associative containers
too, providing a cleaner solution. For ``std::map``, the predicate receives a
``std::pair<const Key, Value>&``.

.. code-block:: cpp

    #include <map>
    #include <string>

    int main() {
      std::map<int, std::string> m{{1, "a"}, {2, "b"}, {3, "c"}};

      // pre-C++20: iterate and erase
      for (auto it = m.begin(); it != m.end();) {
        if (it->first > 1) {
          it = m.erase(it);
        } else {
          ++it;
        }
      }

      // C++20: std::erase_if works on maps too
      // std::erase_if(m, [](auto& p) { return p.first > 1; });
    }

Generating
----------

``std::generate``
~~~~~~~~~~~~~~~~~

:Source: `src/algorithm/generate <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/algorithm/generate>`_

``std::generate`` fills a range by repeatedly calling a generator function and
assigning the returned values to successive elements. The generator can be stateful
(using captured variables or member state) to produce sequences like incrementing
numbers, random values, or computed series. Unlike ``std::fill`` which assigns the
same value to all elements, ``std::generate`` can produce different values for each
position. This is useful for initializing containers with computed values, generating
test data, or creating sequences that depend on external state.

.. code-block:: cpp

    #include <algorithm>
    #include <random>
    #include <vector>

    int main() {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> dist(1, 100);

      std::vector<int> v(5);
      std::generate(v.begin(), v.end(), [&] { return dist(gen); });

      // generate sequence: 0, 1, 2, 3, ...
      std::vector<int> seq(5);
      int n = 0;
      std::generate(seq.begin(), seq.end(), [&n] { return n++; });
    }

``std::iota``
~~~~~~~~~~~~~

:Source: `src/algorithm/iota <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/algorithm/iota>`_

``std::iota`` (from ``<numeric>``) fills a range with sequentially increasing values
starting from a specified initial value. Each element is assigned the previous value
plus one (using ``operator++``). This is more concise than ``std::generate`` with a
counter for simple incrementing sequences. The name comes from the APL programming
language where iota (ι) represents an index generator. Common uses include creating
index arrays, initializing test data, and generating sequences for algorithms that
need position information.

.. code-block:: cpp

    #include <iostream>
    #include <numeric>
    #include <vector>

    int main() {
      std::vector<int> v(5);
      std::iota(v.begin(), v.end(), 1);  // v: {1, 2, 3, 4, 5}

      for (int x : v) std::cout << x << " ";
    }

Copying
-------

``std::copy`` and ``std::copy_if``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Source: `src/algorithm/copy <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/algorithm/copy>`_

``std::copy`` copies elements from a source range to a destination, while
``std::copy_if`` copies only elements satisfying a predicate. The destination
must have sufficient space—use ``std::back_inserter`` to automatically grow the
destination container. These algorithms return an iterator past the last copied
element, useful for chaining operations or determining how many elements were
copied. For overlapping ranges where the destination begins within the source,
use ``std::copy_backward`` instead to avoid overwriting source elements before
they're copied.

.. code-block:: cpp

    #include <algorithm>
    #include <iostream>
    #include <iterator>
    #include <vector>

    int main() {
      std::vector v{1, 2, 3, 4, 5};
      std::vector<int> dest(5);

      std::copy(v.begin(), v.end(), dest.begin());

      // copy to output stream
      std::copy(v.begin(), v.end(), std::ostream_iterator<int>(std::cout, " "));

      // copy_if with predicate
      std::vector<int> evens;
      std::copy_if(v.begin(), v.end(), std::back_inserter(evens),
                   [](int x) { return x % 2 == 0; });
    }

Min/Max
-------

``std::min_element`` and ``std::max_element``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Source: `src/algorithm/minmax <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/algorithm/minmax>`_

``std::min_element`` and ``std::max_element`` return iterators to the smallest and
largest elements in a range, respectively. ``std::minmax_element`` finds both in a
single pass, which is more efficient than calling both separately (approximately
1.5n comparisons vs 2n). These algorithms use ``operator<`` by default but accept
custom comparators. ``std::clamp`` (C++17) constrains a value to a range, returning
the value if within bounds or the nearest bound otherwise. These functions are
essential for range validation, normalization, and finding extrema in datasets.

.. code-block:: cpp

    #include <algorithm>
    #include <iostream>
    #include <vector>

    int main() {
      std::vector v{3, 1, 4, 1, 5, 9, 2, 6};

      auto min_it = std::min_element(v.begin(), v.end());
      auto max_it = std::max_element(v.begin(), v.end());
      auto [lo, hi] = std::minmax_element(v.begin(), v.end());

      std::cout << "Min: " << *min_it << ", Max: " << *max_it << "\n";

      // clamp value to range [lo, hi]
      int val = std::clamp(10, 0, 5);  // returns 5
    }

Checking Conditions
-------------------

``std::all_of``, ``std::any_of``, ``std::none_of``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Source: `src/algorithm/predicates <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/algorithm/predicates>`_

These algorithms test whether elements in a range satisfy a predicate:
``std::all_of`` returns true if **every** element satisfies the predicate,
``std::any_of`` returns true if **at least one** element satisfies it, and
``std::none_of`` returns true if **no** elements satisfy it. They short-circuit
evaluation, stopping as soon as the result is determined. For empty ranges,
``std::all_of`` and ``std::none_of`` return true (vacuous truth), while
``std::any_of`` returns false. These are useful for validation, precondition
checking, and expressing intent more clearly than manual loops.

.. code-block:: cpp

    #include <algorithm>
    #include <iostream>
    #include <vector>

    int main() {
      std::vector v{2, 4, 6, 8};

      bool all_even = std::all_of(v.begin(), v.end(), [](int x) { return x % 2 == 0; });
      bool any_odd = std::any_of(v.begin(), v.end(), [](int x) { return x % 2 != 0; });
      bool none_neg = std::none_of(v.begin(), v.end(), [](int x) { return x < 0; });

      std::cout << std::boolalpha;
      std::cout << "All even: " << all_even << "\n";   // true
      std::cout << "Any odd: " << any_odd << "\n";     // false
      std::cout << "None negative: " << none_neg << "\n";  // true
    }

Unique Elements
---------------

``std::unique``
~~~~~~~~~~~~~~~

:Source: `src/algorithm/unique <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/algorithm/unique>`_

``std::unique`` removes **consecutive** duplicate elements by moving unique elements
to the front and returning an iterator to the new logical end. Like ``std::remove``,
it doesn't actually erase elements—you must call ``erase()`` afterward. To remove
**all** duplicates (not just consecutive ones), sort the range first. The algorithm
compares adjacent elements using ``operator==`` by default, or a custom predicate
for more complex equality definitions. This is commonly used for deduplication
after sorting or for collapsing runs of identical values.

.. code-block:: cpp

    #include <algorithm>
    #include <vector>

    int main() {
      std::vector v{1, 1, 2, 2, 2, 3, 1, 1};

      // remove consecutive duplicates only
      auto last = std::unique(v.begin(), v.end());
      v.erase(last, v.end());
      // v: {1, 2, 3, 1}

      // to remove all duplicates, sort first
      std::vector v2{3, 1, 2, 1, 3, 2};
      std::sort(v2.begin(), v2.end());
      v2.erase(std::unique(v2.begin(), v2.end()), v2.end());
      // v2: {1, 2, 3}
    }

Reversing and Rotating
----------------------

``std::reverse`` and ``std::rotate``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Source: `src/algorithm/reverse-rotate <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/algorithm/reverse-rotate>`_

``std::reverse`` reverses the order of elements in place using bidirectional
iterators. ``std::rotate`` performs a left rotation, moving the element at the
specified middle position to the front while maintaining the relative order of
other elements. Rotation is useful for implementing circular buffers, shifting
elements, or reordering sequences. Both algorithms operate in O(n) time.
``std::rotate`` returns an iterator to the element that was originally at the
beginning, now at its new position after rotation.

.. code-block:: cpp

    #include <algorithm>
    #include <vector>

    int main() {
      std::vector v{1, 2, 3, 4, 5};

      std::reverse(v.begin(), v.end());
      // v: {5, 4, 3, 2, 1}

      std::vector v2{1, 2, 3, 4, 5};
      std::rotate(v2.begin(), v2.begin() + 2, v2.end());
      // v2: {3, 4, 5, 1, 2} - element at position 2 becomes first
    }

Set Operations
--------------

:Source: `src/algorithm/set-operations <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/algorithm/set-operations>`_

Set operations work on **sorted** ranges and compute mathematical set operations:
``std::set_union`` produces elements in either range, ``std::set_intersection``
produces elements in both ranges, ``std::set_difference`` produces elements in
the first range but not the second, and ``std::set_symmetric_difference`` produces
elements in either range but not both. All operations preserve sorted order in
the output and handle duplicates according to their multiplicity. These algorithms
require output iterators and don't modify the input ranges.

.. code-block:: cpp

    #include <algorithm>
    #include <iterator>
    #include <vector>

    int main() {
      std::vector a{1, 2, 3, 4, 5};
      std::vector b{3, 4, 5, 6, 7};

      std::vector<int> result;

      // union: {1, 2, 3, 4, 5, 6, 7}
      std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));

      result.clear();
      // intersection: {3, 4, 5}
      std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                            std::back_inserter(result));

      result.clear();
      // difference (in a but not in b): {1, 2}
      std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                          std::back_inserter(result));
    }

``std::map`` with Custom Comparator
-----------------------------------

:Source: `src/algorithm/map-comparator <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/algorithm/map-comparator>`_

``std::map`` maintains keys in sorted order using ``std::less<Key>`` by default,
which sorts in ascending order. To change the ordering, provide a different
comparator as the third template parameter. ``std::greater<>`` sorts in descending
order. The comparator must define a strict weak ordering (irreflexive, asymmetric,
transitive). Using transparent comparators (like ``std::less<>``) enables
heterogeneous lookup, allowing searches with types convertible to the key type
without creating temporary key objects.

.. code-block:: cpp

    #include <iostream>
    #include <map>

    int main() {
      // ascending (default)
      std::map<int, int, std::less<>> asc{{3, 3}, {1, 1}, {2, 2}};

      // descending
      std::map<int, int, std::greater<>> desc{{3, 3}, {1, 1}, {2, 2}};

      for (auto &[k, v] : asc) std::cout << k << " ";   // 1 2 3
      std::cout << "\n";
      for (auto &[k, v] : desc) std::cout << k << " ";  // 3 2 1
    }

``std::map`` with Object as Key
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Source: `src/algorithm/map-object-key <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/algorithm/map-object-key>`_

To use custom objects as map keys, you must provide a comparison function that
defines a strict weak ordering. This can be a lambda, function object, or
``operator<`` member/free function. The comparator determines both the ordering
and equality (two keys are equal if neither is less than the other). For complex
keys, consider implementing ``operator<=>`` (C++20) which automatically provides
all comparison operators. Alternatively, for unordered containers, implement
``std::hash`` and ``operator==`` instead.

.. code-block:: cpp

    #include <iostream>
    #include <map>

    struct Point {
      int x, y;
    };

    int main() {
      // lambda comparator
      auto cmp = [](const Point &a, const Point &b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
      };

      std::map<Point, int, decltype(cmp)> m(cmp);
      m[{1, 2}] = 1;
      m[{3, 4}] = 2;

      for (auto &[k, v] : m) {
        std::cout << "(" << k.x << "," << k.y << "): " << v << "\n";
      }
    }

C++20 Ranges
------------

C++20 introduced the Ranges library which provides range-based versions of
standard algorithms in the ``std::ranges`` namespace. These accept ranges
directly instead of iterator pairs, reducing boilerplate and potential for
errors. For comprehensive coverage of C++20 ranges, views, and range adaptors,
see :doc:`cpp_iterator`.

.. code-block:: cpp

    #include <algorithm>
    #include <iostream>
    #include <vector>

    int main() {
      std::vector v{5, 3, 1, 4, 2};

      // Range-based sort - no need for begin/end
      std::ranges::sort(v);

      // Range-based find
      auto it = std::ranges::find(v, 3);

      // Range-based predicates
      bool all_pos = std::ranges::all_of(v, [](int x) { return x > 0; });
    }
