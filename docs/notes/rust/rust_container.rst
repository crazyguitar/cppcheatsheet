===========
Collections
===========

.. meta::
   :description: Rust collections compared to C++ STL containers. Covers Vec, HashMap, BTreeMap, HashSet, VecDeque with C++ equivalents.
   :keywords: Rust, Vec, HashMap, BTreeMap, HashSet, VecDeque, collections, C++ STL, vector, map, set

.. contents:: Table of Contents
    :backlinks: none

:Source: `src/rust/collections <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/collections>`_

Rust's standard library provides collections similar to C++ STL containers, but with
a key difference: Rust collections integrate deeply with the ownership and borrowing
system. This integration prevents iterator invalidation bugs at compile time - a
common source of undefined behavior in C++. When you iterate over a Rust collection,
the borrow checker ensures you cannot accidentally modify the collection in ways that
would invalidate the iteration. Additionally, Rust collections are designed with
modern CPU architectures in mind, favoring cache-friendly data structures like
``Vec`` and ``HashMap`` over pointer-heavy structures like linked lists.

The table below maps common C++ STL containers to their Rust equivalents. Note that
Rust's ``HashMap`` and ``HashSet`` use a different hashing algorithm (SipHash by
default) than most C++ implementations, prioritizing security against hash flooding
attacks over raw speed.

+------------------------+------------------------+
| C++                    | Rust                   |
+========================+========================+
| ``std::vector``        | ``Vec<T>``             |
+------------------------+------------------------+
| ``std::map``           | ``BTreeMap<K, V>``     |
+------------------------+------------------------+
| ``std::unordered_map`` | ``HashMap<K, V>``      |
+------------------------+------------------------+
| ``std::set``           | ``BTreeSet<T>``        |
+------------------------+------------------------+
| ``std::unordered_set`` | ``HashSet<T>``         |
+------------------------+------------------------+
| ``std::deque``         | ``VecDeque<T>``        |
+------------------------+------------------------+
| ``std::list``          | ``LinkedList<T>``      |
+------------------------+------------------------+

Vec (Vector)
------------

``Vec<T>`` is Rust's growable array type, equivalent to C++'s ``std::vector``. It
stores elements contiguously in heap memory and provides O(1) amortized push/pop
at the end. The ``vec!`` macro provides convenient initialization syntax similar
to C++'s initializer lists.

The following example demonstrates basic vector operations including creation,
modification, and iteration. Note that Rust requires explicit mutability with
``let mut`` to modify the vector:

**C++:**

.. code-block:: cpp

    #include <iostream>
    #include <vector>

    int main() {
      // Create and initialize a vector
      std::vector<int> v = {1, 2, 3};

      // Add and remove elements from the back
      v.push_back(4);
      v.push_back(5);
      v.pop_back();  // removes 5

      // Iterate and print elements
      std::cout << "Elements: ";
      for (int x : v) {
        std::cout << x << " ";
      }
      std::cout << "\n";

      // Access elements - unchecked vs bounds-checked
      int first = v[0];        // no bounds check, UB if out of range
      int safe = v.at(0);      // throws std::out_of_range if invalid

      std::cout << "First element: " << first << "\n";
      std::cout << "Size: " << v.size() << "\n";

      return 0;
    }

**Rust:**

.. code-block:: rust

    fn main() {
        // Create and initialize a vector using vec! macro
        let mut v = vec![1, 2, 3];

        // Add and remove elements from the back
        v.push(4);
        v.push(5);
        v.pop();  // removes 5, returns Some(5)

        // Iterate and print elements (borrowing with &v)
        print!("Elements: ");
        for x in &v {
            print!("{} ", x);
        }
        println!();

        // Access elements - panics vs returns Option
        let first = v[0];           // panics if out of bounds
        let safe = v.get(0);        // returns Option<&T>, None if invalid

        println!("First element: {}", first);
        println!("Safe access: {:?}", safe);
        println!("Size: {}", v.len());
    }

Vec Operations
~~~~~~~~~~~~~~

This example shows common vector operations including element access, modification,
searching, sorting, and capacity management. Rust's iterator-based search methods
return ``Option`` types, making it explicit when an element might not be found:

**C++:**

.. code-block:: cpp

    #include <algorithm>
    #include <iostream>
    #include <vector>

    int main() {
      std::vector<int> v = {1, 2, 3, 4, 5};

      // Access elements
      int first = v.front();
      int last = v.back();
      // Slice via iterators (no direct slice syntax)

      // Modification
      v.push_back(6);
      v.pop_back();
      v.insert(v.begin(), 0);      // insert at front
      v.erase(v.begin());          // remove from front

      // Search
      auto it = std::find(v.begin(), v.end(), 3);
      bool found = (it != v.end());
      size_t pos = std::distance(v.begin(), it);

      // Sorting
      std::sort(v.begin(), v.end());
      std::sort(v.begin(), v.end(), std::greater<int>());  // reverse

      // Capacity
      size_t len = v.size();
      size_t cap = v.capacity();
      v.reserve(100);
      v.shrink_to_fit();

      std::cout << "Length: " << len << ", Capacity after reserve: " << v.capacity() << "\n";

      return 0;
    }

**Rust:**

.. code-block:: rust

    fn main() {
        let mut v = vec![1, 2, 3, 4, 5];

        // Access elements
        let first = &v[0];
        let last = v.last();              // Option<&T>
        let slice = &v[1..3];             // [2, 3] - native slice syntax
        println!("First: {}, Last: {:?}, Slice: {:?}", first, last, slice);

        // Modification
        v.push(6);
        v.pop();                          // returns Option<T>
        v.insert(0, 0);                   // insert at index
        v.remove(0);                      // remove at index, returns element

        // Search - returns Option types for safety
        let pos = v.iter().position(|&x| x == 3);  // Option<usize>
        let contains = v.contains(&3);             // bool
        println!("Position of 3: {:?}, Contains 3: {}", pos, contains);

        // Sorting
        v.sort();
        v.sort_by(|a, b| b.cmp(a));       // reverse order

        // Capacity management
        let len = v.len();
        let cap = v.capacity();
        v.reserve(100);
        v.shrink_to_fit();

        println!("Length: {}, Capacity: {}", len, v.capacity());
    }

HashMap
-------

``HashMap<K, V>`` is Rust's hash table implementation, equivalent to C++'s
``std::unordered_map``. Keys must implement the ``Hash`` and ``Eq`` traits. The
``entry`` API provides an elegant way to handle insert-or-update patterns that
would require multiple lookups in C++.

This example demonstrates basic hash map operations including insertion, lookup,
and iteration. Note that Rust's ``get`` method returns an ``Option``, making it
impossible to accidentally use a non-existent key:

**C++:**

.. code-block:: cpp

    #include <iostream>
    #include <string>
    #include <unordered_map>

    int main() {
      std::unordered_map<std::string, int> map;

      // Insert elements
      map["one"] = 1;
      map["two"] = 2;
      map.insert({"three", 3});

      // Lookup - must check if key exists
      if (auto it = map.find("one"); it != map.end()) {
        std::cout << "Found: " << it->second << "\n";
      }

      // Dangerous: creates entry if missing!
      int val = map["four"];  // inserts {"four", 0}

      // Iterate over key-value pairs
      std::cout << "All entries:\n";
      for (const auto& [key, value] : map) {
        std::cout << "  " << key << ": " << value << "\n";
      }

      std::cout << "Size: " << map.size() << "\n";

      return 0;
    }

**Rust:**

.. code-block:: rust

    use std::collections::HashMap;

    fn main() {
        let mut map = HashMap::new();

        // Insert elements
        map.insert("one", 1);
        map.insert("two", 2);
        map.insert("three", 3);

        // Lookup - returns Option, no accidental insertion
        if let Some(value) = map.get("one") {
            println!("Found: {}", value);
        }

        // Safe: get returns None for missing keys
        let missing = map.get("four");
        println!("Missing key: {:?}", missing);  // None

        // Iterate over key-value pairs
        println!("All entries:");
        for (key, value) in &map {
            println!("  {}: {}", key, value);
        }

        println!("Size: {}", map.len());
    }

HashMap Entry API
~~~~~~~~~~~~~~~~~

The entry API is one of Rust's most powerful collection features. It allows you to
inspect and modify map entries in a single lookup, avoiding the double-lookup
pattern common in C++. This is particularly useful for counting, caching, and
accumulation patterns:

**C++:**

.. code-block:: cpp

    #include <iostream>
    #include <string>
    #include <unordered_map>

    int main() {
      std::unordered_map<std::string, int> word_count;
      std::string words[] = {"apple", "banana", "apple", "cherry", "banana", "apple"};

      // Count words - requires lookup then insert/update
      for (const auto& word : words) {
        auto it = word_count.find(word);
        if (it != word_count.end()) {
          it->second++;
        } else {
          word_count[word] = 1;
        }
      }

      // Or use operator[] which default-constructs if missing
      std::unordered_map<std::string, int> word_count2;
      for (const auto& word : words) {
        word_count2[word]++;  // relies on int default being 0
      }

      for (const auto& [word, count] : word_count) {
        std::cout << word << ": " << count << "\n";
      }

      return 0;
    }

**Rust:**

.. code-block:: rust

    use std::collections::HashMap;

    fn main() {
        let mut word_count = HashMap::new();
        let words = ["apple", "banana", "apple", "cherry", "banana", "apple"];

        // Count words using entry API - single lookup!
        for word in &words {
            // or_insert returns &mut V, allowing in-place modification
            *word_count.entry(*word).or_insert(0) += 1;
        }

        // Entry API variants:
        let mut map: HashMap<&str, Vec<i32>> = HashMap::new();

        // or_insert: insert value if missing
        map.entry("numbers").or_insert(vec![]).push(1);

        // or_insert_with: insert computed value if missing (lazy)
        map.entry("computed").or_insert_with(|| {
            println!("Computing default...");
            vec![42]
        });

        // or_default: insert Default::default() if missing
        map.entry("default").or_default().push(100);

        // Print word counts
        println!("Word counts:");
        for (word, count) in &word_count {
            println!("  {}: {}", word, count);
        }
    }

BTreeMap (Ordered Map)
----------------------

``BTreeMap<K, V>`` is Rust's ordered map, equivalent to C++'s ``std::map``. It keeps
keys in sorted order and provides O(log n) operations. Unlike ``HashMap``, keys only
need to implement ``Ord`` (not ``Hash``). BTreeMap also supports efficient range
queries, which are not available with hash-based maps:

**C++:**

.. code-block:: cpp

    #include <iostream>
    #include <map>
    #include <string>

    int main() {
      std::map<int, std::string> map;

      // Insert in arbitrary order
      map[3] = "three";
      map[1] = "one";
      map[2] = "two";
      map[5] = "five";
      map[4] = "four";

      // Iteration is in sorted key order
      std::cout << "All entries (sorted):\n";
      for (const auto& [key, value] : map) {
        std::cout << "  " << key << ": " << value << "\n";
      }

      // Range query using lower_bound/upper_bound
      std::cout << "Range [2, 4):\n";
      for (auto it = map.lower_bound(2); it != map.lower_bound(4); ++it) {
        std::cout << "  " << it->first << ": " << it->second << "\n";
      }

      return 0;
    }

**Rust:**

.. code-block:: rust

    use std::collections::BTreeMap;

    fn main() {
        let mut map = BTreeMap::new();

        // Insert in arbitrary order
        map.insert(3, "three");
        map.insert(1, "one");
        map.insert(2, "two");
        map.insert(5, "five");
        map.insert(4, "four");

        // Iteration is in sorted key order
        println!("All entries (sorted):");
        for (k, v) in &map {
            println!("  {}: {}", k, v);
        }

        // Range queries - more ergonomic than C++
        println!("Range [2, 4):");
        for (k, v) in map.range(2..4) {
            println!("  {}: {}", k, v);
        }

        // Other range variants
        println!("Keys >= 3:");
        for (k, v) in map.range(3..) {
            println!("  {}: {}", k, v);
        }
    }

HashSet
-------

``HashSet<T>`` is Rust's hash set implementation, equivalent to C++'s
``std::unordered_set``. It stores unique values and provides O(1) average-case
lookup, insertion, and removal. Rust's HashSet provides convenient methods for
set-theoretic operations like union, intersection, and difference:

**C++:**

.. code-block:: cpp

    #include <iostream>
    #include <unordered_set>
    #include <algorithm>
    #include <iterator>

    int main() {
      std::unordered_set<int> set = {1, 2, 3};

      // Insert and check membership
      set.insert(4);
      set.insert(2);  // duplicate, ignored
      bool has_two = set.count(2) > 0;

      std::cout << "Contains 2: " << (has_two ? "yes" : "no") << "\n";
      std::cout << "Size: " << set.size() << "\n";

      // Set operations require manual implementation or std::set_*
      std::unordered_set<int> other = {3, 4, 5};

      // No built-in union/intersection for unordered_set
      // Must convert to sorted containers or implement manually

      return 0;
    }

**Rust:**

.. code-block:: rust

    use std::collections::HashSet;

    fn main() {
        let mut set: HashSet<i32> = [1, 2, 3].into();

        // Insert and check membership
        set.insert(4);
        set.insert(2);  // duplicate, returns false
        let has_two = set.contains(&2);

        println!("Contains 2: {}", has_two);
        println!("Size: {}", set.len());

        // Iterate (order is arbitrary)
        print!("Elements: ");
        for x in &set {
            print!("{} ", x);
        }
        println!();
    }

Set Operations
~~~~~~~~~~~~~~

Rust provides built-in methods for common set operations. These methods return
iterators, allowing lazy evaluation and chaining. The following example demonstrates
union, intersection, difference, and symmetric difference:

**C++:**

.. code-block:: cpp

    #include <algorithm>
    #include <iostream>
    #include <set>  // using ordered set for set operations
    #include <iterator>

    int main() {
      std::set<int> a = {1, 2, 3};
      std::set<int> b = {2, 3, 4};

      // Union
      std::set<int> union_set;
      std::set_union(a.begin(), a.end(), b.begin(), b.end(),
                     std::inserter(union_set, union_set.begin()));

      // Intersection
      std::set<int> intersection;
      std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                            std::inserter(intersection, intersection.begin()));

      // Difference (a - b)
      std::set<int> difference;
      std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                          std::inserter(difference, difference.begin()));

      // Symmetric difference
      std::set<int> sym_diff;
      std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(),
                                    std::inserter(sym_diff, sym_diff.begin()));

      auto print_set = [](const std::string& name, const std::set<int>& s) {
        std::cout << name << ": {";
        for (auto it = s.begin(); it != s.end(); ++it) {
          if (it != s.begin()) std::cout << ", ";
          std::cout << *it;
        }
        std::cout << "}\n";
      };

      print_set("Union", union_set);
      print_set("Intersection", intersection);
      print_set("Difference", difference);
      print_set("Symmetric Diff", sym_diff);

      return 0;
    }

**Rust:**

.. code-block:: rust

    use std::collections::HashSet;

    fn main() {
        let a: HashSet<i32> = [1, 2, 3].into();
        let b: HashSet<i32> = [2, 3, 4].into();

        // Set operations return iterators - collect into HashSet
        let union_set: HashSet<_> = a.union(&b).cloned().collect();
        let intersection: HashSet<_> = a.intersection(&b).cloned().collect();
        let difference: HashSet<_> = a.difference(&b).cloned().collect();
        let sym_diff: HashSet<_> = a.symmetric_difference(&b).cloned().collect();

        println!("Union: {:?}", union_set);           // {1, 2, 3, 4}
        println!("Intersection: {:?}", intersection); // {2, 3}
        println!("Difference: {:?}", difference);     // {1}
        println!("Symmetric Diff: {:?}", sym_diff);   // {1, 4}

        // Predicates for set relationships
        let subset: HashSet<i32> = [2, 3].into();
        println!("Is {{2,3}} subset of a? {}", subset.is_subset(&a));
        println!("Are a and b disjoint? {}", a.is_disjoint(&b));
    }

VecDeque (Double-ended Queue)
-----------------------------

``VecDeque<T>`` is Rust's double-ended queue, equivalent to C++'s ``std::deque``.
It provides O(1) push and pop operations at both ends, making it ideal for
queue and sliding window implementations. Unlike ``Vec``, it uses a ring buffer
internally:

**C++:**

.. code-block:: cpp

    #include <deque>
    #include <iostream>

    int main() {
      std::deque<int> dq = {2, 3, 4};

      // Push to both ends
      dq.push_front(1);
      dq.push_back(5);

      // Pop from both ends
      dq.pop_front();
      dq.pop_back();

      // Access elements
      std::cout << "Front: " << dq.front() << "\n";
      std::cout << "Back: " << dq.back() << "\n";

      // Random access
      std::cout << "Element at index 1: " << dq[1] << "\n";

      // Iterate
      std::cout << "Elements: ";
      for (int x : dq) {
        std::cout << x << " ";
      }
      std::cout << "\n";

      return 0;
    }

**Rust:**

.. code-block:: rust

    use std::collections::VecDeque;

    fn main() {
        let mut dq = VecDeque::from([2, 3, 4]);

        // Push to both ends
        dq.push_front(1);
        dq.push_back(5);

        // Pop from both ends - returns Option<T>
        let front = dq.pop_front();  // Some(1)
        let back = dq.pop_back();    // Some(5)
        println!("Popped front: {:?}, back: {:?}", front, back);

        // Access elements
        println!("Front: {:?}", dq.front());  // Option<&T>
        println!("Back: {:?}", dq.back());

        // Random access
        println!("Element at index 1: {:?}", dq.get(1));

        // Iterate
        print!("Elements: ");
        for x in &dq {
            print!("{} ", x);
        }
        println!();
    }

Iterator Invalidation
---------------------

One of Rust's most significant safety improvements over C++ is compile-time
prevention of iterator invalidation. In C++, modifying a collection while
iterating over it leads to undefined behavior that may silently corrupt data
or crash. Rust's borrow checker makes this a compile-time error:

**C++ (compiles but has undefined behavior):**

.. code-block:: cpp

    #include <iostream>
    #include <vector>

    int main() {
      std::vector<int> v = {1, 2, 3, 4, 5};

      // DANGEROUS: modifying vector while iterating
      for (auto it = v.begin(); it != v.end(); ++it) {
        if (*it == 3) {
          v.erase(it);  // iterator invalidated! UB!
          // Even "fixing" with: it = v.erase(it); --it;
          // is error-prone
        }
      }

      // This might crash, print garbage, or appear to work
      for (int x : v) {
        std::cout << x << " ";
      }

      return 0;
    }

**Rust (compile error prevents the bug):**

.. code-block:: rust

    fn main() {
        let mut v = vec![1, 2, 3, 4, 5];

        // This won't compile - borrow checker prevents it
        // for x in &v {
        //     if *x == 3 {
        //         v.remove(2);  // error: cannot borrow `v` as mutable
        //     }                 // because it's borrowed as immutable
        // }

        // Correct approach 1: use retain (most idiomatic)
        v.retain(|&x| x != 3);
        println!("After retain: {:?}", v);

        // Correct approach 2: collect indices first, remove in reverse
        let mut v2 = vec![1, 2, 3, 4, 5];
        let indices: Vec<_> = v2.iter()
            .enumerate()
            .filter(|(_, &x)| x == 3)
            .map(|(i, _)| i)
            .collect();
        for i in indices.into_iter().rev() {
            v2.remove(i);
        }
        println!("After index removal: {:?}", v2);

        // Correct approach 3: drain_filter (nightly) or filter + collect
        let v3 = vec![1, 2, 3, 4, 5];
        let v3: Vec<_> = v3.into_iter().filter(|&x| x != 3).collect();
        println!("After filter: {:?}", v3);
    }

See Also
--------

- :doc:`rust_iterator` - Iterator adapters and patterns
- :doc:`rust_basic` - Ownership with collections
