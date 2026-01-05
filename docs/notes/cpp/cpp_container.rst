=========
Container
=========

.. contents:: Table of Contents
    :backlinks: none

std::vector
-----------

:Source: `src/container/vector <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/container/vector>`_

``std::vector`` is a dynamic array that provides fast random access and efficient
insertion/removal at the end. It is the most commonly used container in C++ due
to its cache-friendly memory layout and versatility. Elements are stored in
contiguous memory, which means pointer arithmetic works and the underlying array
can be passed to C APIs via ``data()``.

The vector manages its own memory and automatically grows when needed. When
capacity is exceeded, it allocates a new larger buffer (typically 1.5x or 2x
the current capacity), copies/moves all elements, and deallocates the old buffer.
This reallocation invalidates all iterators and references to elements.

**Pros:**

- O(1) random access by index
- O(1) amortized ``push_back()`` and ``pop_back()``
- Cache-friendly contiguous memory layout
- Compatible with C APIs via ``data()``
- Most memory-efficient for storing elements (no per-element overhead)

**Cons:**

- O(n) insertion/removal at front or middle (elements must be shifted)
- Reallocation invalidates all iterators and references
- Growing may cause expensive copy/move of all elements
- ``push_front()`` not available (would be O(n))

**Initialization:**

.. code-block:: cpp

    #include <vector>

    int main(int argc, char *argv[]) {
      std::vector<int> v1;                // Empty vector
      std::vector<int> v2(5);             // 5 elements, value-initialized (0)
      std::vector<int> v3(5, 10);         // 5 elements, all 10
      std::vector<int> v4{1, 2, 3, 4, 5}; // Initializer list
      std::vector<int> v5(v4);            // Copy constructor
      std::vector<int> v6(v4.begin(), v4.end()); // Range constructor
    }

**Element access:**

.. code-block:: cpp

    #include <iostream>
    #include <vector>

    int main(int argc, char *argv[]) {
      std::vector<int> v{1, 2, 3, 4, 5};

      std::cout << v[0] << "\n";      // No bounds checking
      std::cout << v.at(0) << "\n";   // Throws std::out_of_range if invalid
      std::cout << v.front() << "\n"; // First element
      std::cout << v.back() << "\n";  // Last element
      std::cout << v.data()[0] << "\n"; // Raw pointer access
    }

**Modifiers:**

.. code-block:: cpp

    #include <vector>

    int main(int argc, char *argv[]) {
      std::vector<int> v;

      v.push_back(1);              // Add to end
      v.emplace_back(2);           // Construct in place at end
      v.pop_back();                // Remove from end
      v.insert(v.begin(), 0);      // Insert at position
      v.erase(v.begin());          // Remove at position
      v.clear();                   // Remove all elements
    }

**Capacity management:**

Understanding capacity vs size is crucial for performance. ``size()`` is the
number of elements, while ``capacity()`` is the allocated storage. Use
``reserve()`` to pre-allocate memory when you know the approximate size to
avoid repeated reallocations:

.. code-block:: cpp

    #include <iostream>
    #include <vector>

    int main(int argc, char *argv[]) {
      std::vector<int> v;

      v.reserve(100);  // Pre-allocate memory for 100 elements
      std::cout << v.capacity() << "\n";  // 100
      std::cout << v.size() << "\n";      // 0

      v.resize(50);    // Change size to 50 (value-initialized)
      std::cout << v.size() << "\n";      // 50

      v.shrink_to_fit();  // Request to reduce capacity to fit size
    }

std::deque
----------

:Source: `src/container/deque <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/container/deque>`_

``std::deque`` (double-ended queue) provides fast insertion and removal at both
ends. Unlike ``std::vector``, it is implemented as a sequence of fixed-size
arrays (chunks), with a central map keeping track of the chunks. This design
allows O(1) operations at both front and back without moving existing elements.

The deque provides random access like vector, but with slightly more overhead
due to the indirection through the chunk map. Elements are not stored in fully
contiguous memory, so ``data()`` is not available and cache performance may be
slightly worse than vector for sequential access.

**Pros:**

- O(1) ``push_front()``, ``pop_front()``, ``push_back()``, ``pop_back()``
- O(1) random access by index (slightly slower than vector)
- No reallocation of existing elements when growing
- Iterators to non-erased elements remain valid after insertions at ends

**Cons:**

- Higher memory overhead than vector (chunk management)
- Not contiguous; no ``data()`` member function
- Slightly slower random access due to indirection
- O(n) insertion/removal in the middle

.. code-block:: cpp

    #include <deque>
    #include <iostream>

    int main(int argc, char *argv[]) {
      std::deque<int> d{2, 3, 4};

      d.push_front(1);  // Add to front: {1, 2, 3, 4}
      d.push_back(5);   // Add to back:  {1, 2, 3, 4, 5}
      d.pop_front();    // Remove front: {2, 3, 4, 5}
      d.pop_back();     // Remove back:  {2, 3, 4}

      std::cout << d.front() << "\n";  // 2
      std::cout << d.back() << "\n";   // 4
      std::cout << d[1] << "\n";       // 3 (random access)
    }

std::list
---------

:Source: `src/container/list <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/container/list>`_

``std::list`` is a doubly-linked list where each element is stored in a separate
node containing the data and pointers to the previous and next nodes. This
structure allows O(1) insertion and removal at any position given an iterator,
but sacrifices random access and cache locality.

The list is the only standard sequence container that provides ``splice()``,
which can transfer elements between lists in O(1) time without copying or
moving the actual elements. It also provides ``merge()``, ``sort()``, and
``unique()`` as member functions optimized for linked list operations.

**Pros:**

- O(1) insertion/removal anywhere given an iterator
- Iterators and references never invalidated (except for erased elements)
- O(1) ``splice()`` to transfer elements between lists
- Stable addresses; elements never move in memory

**Cons:**

- No random access; must traverse from beginning or end
- O(n) to access element by index
- Poor cache locality; nodes scattered in memory
- High memory overhead (two pointers per element)
- Slower iteration than vector due to pointer chasing

.. code-block:: cpp

    #include <iostream>
    #include <list>

    int main(int argc, char *argv[]) {
      std::list<int> l{1, 2, 3};

      l.push_front(0);   // {0, 1, 2, 3}
      l.push_back(4);    // {0, 1, 2, 3, 4}

      auto it = l.begin();
      std::advance(it, 2);
      l.insert(it, 99);  // {0, 1, 99, 2, 3, 4}

      l.remove(99);      // Remove all elements equal to 99
      l.reverse();       // Reverse the list in O(n)
      l.sort();          // Sort the list in O(n log n)

      // Splice: transfer elements from another list
      std::list<int> other{10, 20};
      l.splice(l.end(), other);  // other is now empty

      for (const auto &x : l) {
        std::cout << x << " ";
      }
      std::cout << "\n";
    }

std::set
--------

:Source: `src/container/set <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/container/set>`_

``std::set`` is an ordered associative container that stores unique elements.
It is typically implemented as a self-balancing binary search tree (red-black tree),
which guarantees O(log n) time complexity for insertion, deletion, and lookup
operations regardless of the input order.

The container automatically maintains elements in sorted order according to the
comparison function (default is ``std::less<T>``). This makes it ideal for
scenarios where you need to maintain a sorted collection with fast membership
testing, or when you need to perform range-based queries like finding all
elements between two values.

**Pros:**

- Elements are always sorted, enabling efficient range queries with ``lower_bound()`` and ``upper_bound()``
- Guaranteed O(log n) worst-case performance for all operations
- No duplicates allowed, automatically enforced
- Iterators remain valid after insertions (except for erased elements)

**Cons:**

- O(log n) operations are slower than O(1) hash-based containers for simple lookups
- Higher memory overhead due to tree node pointers (typically 3 pointers per element)
- Poor cache locality since nodes are scattered in memory
- Requires elements to be comparable with ``<`` operator

.. code-block:: cpp

    #include <iostream>
    #include <set>

    int main(int argc, char *argv[]) {
      std::set<int> s{3, 1, 4, 1, 5};  // Duplicates ignored: {1, 3, 4, 5}

      s.insert(2);                     // {1, 2, 3, 4, 5}
      s.erase(3);                      // {1, 2, 4, 5}

      if (s.find(2) != s.end()) {
        std::cout << "Found 2\n";
      }

      if (s.contains(4)) {             // C++20
        std::cout << "Contains 4\n";
      }

      // Range query: find elements in [2, 5)
      auto lo = s.lower_bound(2);
      auto hi = s.lower_bound(5);
      for (auto it = lo; it != hi; ++it) {
        std::cout << *it << " ";       // Output: 2 4
      }
      std::cout << "\n";
    }

std::unordered_set
------------------

:Source: `src/container/unordered-set <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/container/unordered-set>`_

``std::unordered_set`` is a hash table-based container that stores unique elements.
It uses a hash function to map elements to buckets, providing O(1) average-case
time complexity for insertion, deletion, and lookup operations.

The container does not maintain any particular order of elements. The actual
order depends on the hash function and the internal bucket structure, which
may change when the container is rehashed (typically when the load factor
exceeds a threshold). This makes it unsuitable for scenarios requiring
ordered iteration.

**Pros:**

- O(1) average-case operations make it significantly faster than ``std::set`` for large datasets
- Excellent performance when hash function distributes elements evenly
- Simple membership testing with ``count()`` or ``contains()`` (C++20)

**Cons:**

- Elements are not sorted; iteration order is unspecified and may change
- O(n) worst-case when many elements hash to the same bucket (hash collisions)
- Requires elements to be hashable (``std::hash<T>`` must be defined) and equality comparable
- Higher memory usage due to hash table overhead (bucket array, load factor management)
- Iterators may be invalidated by insertions that trigger rehashing

.. code-block:: cpp

    #include <iostream>
    #include <unordered_set>

    int main(int argc, char *argv[]) {
      std::unordered_set<int> s{3, 1, 4, 1, 5};

      s.insert(2);
      s.erase(3);

      if (s.find(2) != s.end()) {
        std::cout << "Found 2\n";
      }

      // Check load factor and bucket count
      std::cout << "Load factor: " << s.load_factor() << "\n";
      std::cout << "Bucket count: " << s.bucket_count() << "\n";

      // Iteration order is unspecified
      for (const auto &x : s) {
        std::cout << x << " ";
      }
      std::cout << "\n";
    }

std::map
--------

:Source: `src/container/map <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/container/map>`_

``std::map`` is an ordered associative container that stores key-value pairs
with unique keys. Like ``std::set``, it is typically implemented as a red-black
tree, providing O(log n) operations. The elements are sorted by keys according
to the comparison function.

The ``operator[]`` provides convenient access but has a subtle behavior: if the
key doesn't exist, it inserts a default-constructed value. Use ``find()`` or
``at()`` when you don't want to accidentally insert elements. The ``at()``
method throws ``std::out_of_range`` if the key is not found.

**Pros:**

- Keys are always sorted, enabling efficient range queries and ordered iteration
- Guaranteed O(log n) worst-case performance
- ``lower_bound()`` and ``upper_bound()`` enable efficient range-based access
- Stable iterators (not invalidated by insertions)

**Cons:**

- O(log n) operations are slower than hash-based containers for simple lookups
- ``operator[]`` silently inserts default values for missing keys
- Higher memory overhead due to tree structure
- Poor cache locality compared to contiguous containers

.. code-block:: cpp

    #include <iostream>
    #include <map>
    #include <string>

    int main(int argc, char *argv[]) {
      std::map<std::string, int> m{{"apple", 1}, {"banana", 2}};

      m["cherry"] = 3;                 // Insert or update
      m.insert({"date", 4});           // Insert only if key doesn't exist
      m.erase("banana");

      // Safe access without inserting
      if (auto it = m.find("apple"); it != m.end()) {
        std::cout << "apple: " << it->second << "\n";
      }

      // at() throws if key not found
      try {
        std::cout << m.at("grape") << "\n";
      } catch (const std::out_of_range &e) {
        std::cout << "Key not found\n";
      }

      for (const auto &[key, value] : m) {  // Structured binding (C++17)
        std::cout << key << ": " << value << "\n";  // Sorted by key
      }
    }

std::unordered_map
------------------

:Source: `src/container/unordered-map <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/container/unordered-map>`_

``std::unordered_map`` is a hash table-based container that stores key-value
pairs with unique keys. It provides O(1) average-case operations, making it
the fastest associative container for most use cases where ordering is not
required.

The container is widely used for caching, counting occurrences, and building
lookup tables. For custom key types, you must provide both a hash function
(specializing ``std::hash<T>``) and an equality operator.

**Pros:**

- O(1) average-case operations; typically the fastest associative container
- Excellent for frequency counting, caching, and lookup tables
- ``try_emplace()`` (C++17) avoids unnecessary constructions

**Cons:**

- Keys are not sorted; iteration order is unspecified
- O(n) worst-case with poor hash functions or adversarial input
- Requires hashable keys (``std::hash<T>``) and equality comparable
- No range queries or ordered iteration
- Rehashing can invalidate iterators and cause performance spikes

.. code-block:: cpp

    #include <iostream>
    #include <string>
    #include <unordered_map>

    int main(int argc, char *argv[]) {
      std::unordered_map<std::string, int> m{{"apple", 1}, {"banana", 2}};

      m["cherry"] = 3;
      m.insert({"date", 4});
      m.erase("banana");

      // Counting pattern
      std::unordered_map<char, int> freq;
      for (char c : std::string("hello")) {
        freq[c]++;
      }
      std::cout << "l appears " << freq['l'] << " times\n";

      // Safe access
      if (auto it = m.find("apple"); it != m.end()) {
        std::cout << "apple: " << it->second << "\n";
      }

      // Iteration order is unspecified
      for (const auto &[key, value] : m) {
        std::cout << key << ": " << value << "\n";
      }
    }

Ordered vs Unordered Containers
-------------------------------

.. list-table::
   :header-rows: 1

   * - Feature
     - set/map
     - unordered_set/map
   * - Implementation
     - Red-black tree
     - Hash table
   * - Lookup
     - O(log n)
     - O(1) average, O(n) worst
   * - Insert/Delete
     - O(log n)
     - O(1) average, O(n) worst
   * - Ordered iteration
     - Yes
     - No
   * - Key requirement
     - Comparable (<)
     - Hashable + equality (==)
   * - Memory overhead
     - Tree pointers
     - Hash buckets

**When to use ordered (set/map):**

- Need sorted iteration or range queries
- Keys are not easily hashable
- Predictable worst-case performance required

**When to use unordered (unordered_set/map):**

- Performance is critical and dataset is large
- Don't need sorted order
- Keys have good hash distribution

std::priority_queue
-------------------

:Source: `src/container/priority-queue <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/container/priority-queue>`_

``std::priority_queue`` is a container adapter that provides constant time
lookup of the largest (by default) element, with O(log n) insertion and
extraction. It is implemented as a binary heap, typically using ``std::vector``
as the underlying container.

Unlike other containers, ``std::priority_queue`` only provides access to the
top element. You cannot iterate over elements or access them by index. The
heap property ensures the largest element (according to the comparator) is
always at the top, but the remaining elements are not fully sorted.

**Pros:**

- O(1) access to the maximum (or minimum with custom comparator) element
- O(log n) insertion and removal of top element
- Efficient for scenarios requiring repeated access to extreme values
- Useful for algorithms like Dijkstra's shortest path, merge k sorted lists

**Cons:**

- No iteration or random access to elements
- Cannot modify elements other than removing the top
- No ``clear()`` member function (must pop all elements or reassign)
- Underlying container is not directly accessible

**Default max-heap:**

.. code-block:: cpp

    #include <iostream>
    #include <queue>
    #include <vector>

    int main(int argc, char *argv[]) {
      std::vector<int> data{1, 5, 2, 1, 3};
      std::priority_queue<int> pq;

      for (const auto &x : data) {
        pq.push(x);
      }

      while (!pq.empty()) {
        std::cout << pq.top() << " ";  // Output: 5 3 2 1 1
        pq.pop();
      }
      std::cout << "\n";
    }

**Min-heap using std::greater:**

To create a min-heap, use ``std::greater<T>`` as the comparator:

.. code-block:: cpp

    #include <functional>
    #include <iostream>
    #include <queue>
    #include <vector>

    int main(int argc, char *argv[]) {
      std::priority_queue<int, std::vector<int>, std::greater<int>> pq;

      for (int x : {1, 5, 2, 1, 3}) {
        pq.push(x);
      }

      while (!pq.empty()) {
        std::cout << pq.top() << " ";  // Output: 1 1 2 3 5
        pq.pop();
      }
      std::cout << "\n";
    }

**Custom comparator with lambda:**

.. code-block:: cpp

    #include <iostream>
    #include <queue>
    #include <vector>

    int main(int argc, char *argv[]) {
      auto cmp = [](int a, int b) { return a > b; };  // Min-heap
      std::priority_queue<int, std::vector<int>, decltype(cmp)> pq(cmp);

      for (int x : {1, 5, 2, 1, 3}) {
        pq.push(x);
      }

      while (!pq.empty()) {
        std::cout << pq.top() << " ";  // Output: 1 1 2 3 5
        pq.pop();
      }
      std::cout << "\n";
    }

**Merging sorted lists:**

Priority queues are useful for merging multiple sorted sequences:

.. code-block:: cpp

    #include <iostream>
    #include <queue>
    #include <vector>

    int main(int argc, char *argv[]) {
      std::priority_queue<int> pq;
      std::vector<int> list1{9, 7, 8};
      std::vector<int> list2{0, 5, 3};

      for (const auto &x : list1) pq.push(x);
      for (const auto &x : list2) pq.push(x);

      while (!pq.empty()) {
        std::cout << pq.top() << " ";  // Output: 9 8 7 5 3 0
        pq.pop();
      }
      std::cout << "\n";
    }

Container Performance Comparison
--------------------------------

:Source: `src/container/profile <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/container/profile>`_

Different containers have different performance characteristics. The following
table summarizes the time complexity of common operations:

.. list-table::
   :header-rows: 1

   * - Operation
     - vector
     - deque
     - list
   * - push_back
     - O(1)*
     - O(1)
     - O(1)
   * - push_front
     - O(n)
     - O(1)
     - O(1)
   * - pop_back
     - O(1)
     - O(1)
     - O(1)
   * - pop_front
     - O(n)
     - O(1)
     - O(1)
   * - Random access
     - O(1)
     - O(1)
     - O(n)

\* Amortized constant time

.. note::

    See ``src/container/profile`` for benchmark code comparing these operations
    across different container types.
