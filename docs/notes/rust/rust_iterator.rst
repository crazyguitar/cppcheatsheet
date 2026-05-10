=========
Iterators
=========

.. meta::
   :description: Rust iterators compared to C++ STL iterators and ranges. Covers iterator adapters, lazy evaluation, and common patterns.
   :keywords: Rust, iterators, iter, map, filter, collect, fold, C++ STL, ranges

.. contents:: Table of Contents
    :backlinks: none

:Source: `src/rust/iterators <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/iterators>`_

Rust iterators are lazy and chainable, similar to C++20 ranges. Operations like
``map`` and ``filter`` don't execute until you consume the iterator (e.g., with
``collect``). This lazy evaluation allows the compiler to fuse multiple operations
into a single pass over the data, often generating code as efficient as hand-written
loops. Unlike C++ STL algorithms which operate on iterator pairs, Rust iterators are
single objects that carry their own state, making them easier to compose and pass
around. The iterator trait system also enables powerful abstractions - any type that
implements ``Iterator`` automatically gains access to dozens of adapter methods.

Basic Iteration
---------------

Rust provides multiple ways to iterate over collections, each with different
ownership semantics. The ``for`` loop syntax automatically calls ``into_iter()``
on the collection, but you can control borrowing behavior with ``&`` and ``&mut``.
This example shows the basic iteration patterns in both languages:

**C++:**

.. code-block:: cpp

    #include <iostream>
    #include <vector>

    int main() {
      std::vector<int> v = {1, 2, 3, 4, 5};

      // Range-based for (copies by default, use & for reference)
      std::cout << "By value: ";
      for (int x : v) {
        std::cout << x << " ";
      }
      std::cout << "\n";

      // By reference (no copy)
      std::cout << "By reference: ";
      for (const int& x : v) {
        std::cout << x << " ";
      }
      std::cout << "\n";

      // Iterator-based (traditional style)
      std::cout << "Iterator-based: ";
      for (auto it = v.begin(); it != v.end(); ++it) {
        std::cout << *it << " ";
      }
      std::cout << "\n";

      return 0;
    }

**Rust:**

.. code-block:: rust

    fn main() {
        let v = vec![1, 2, 3, 4, 5];

        // Borrowing iteration - v remains valid after loop
        print!("By reference: ");
        for x in &v {
            print!("{} ", x);  // x is &i32
        }
        println!();

        // v is still valid here
        println!("v still has {} elements", v.len());

        // Consuming iteration - v is moved into the loop
        print!("By value (consuming): ");
        for x in v {
            print!("{} ", x);  // x is i32
        }
        println!();

        // v is no longer valid here - would be compile error to use it
        // println!("{}", v.len());  // error: borrow of moved value
    }

Three Iterator Types
--------------------

Rust collections provide three iterator methods that control ownership: ``iter()``
borrows elements immutably, ``iter_mut()`` borrows mutably, and ``into_iter()``
takes ownership. Understanding these is crucial for writing efficient Rust code.
C++ achieves similar patterns through const/non-const iterators and move semantics:

**C++:**

.. code-block:: cpp

    #include <iostream>
    #include <vector>

    int main() {
      std::vector<int> v = {1, 2, 3};

      // Const iteration (like Rust's iter())
      std::cout << "Const iteration: ";
      for (auto it = v.cbegin(); it != v.cend(); ++it) {
        std::cout << *it << " ";
        // *it = 10;  // error: cannot modify through const iterator
      }
      std::cout << "\n";

      // Mutable iteration (like Rust's iter_mut())
      std::cout << "Mutable iteration: ";
      for (auto it = v.begin(); it != v.end(); ++it) {
        *it += 10;  // can modify
        std::cout << *it << " ";
      }
      std::cout << "\n";

      // Move iteration (like Rust's into_iter())
      std::vector<std::string> strings = {"hello", "world"};
      std::vector<std::string> moved;
      for (auto& s : strings) {
        moved.push_back(std::move(s));  // explicit move required
      }
      // strings still exists but elements are in moved-from state

      return 0;
    }

**Rust:**

.. code-block:: rust

    fn main() {
        let mut v = vec![1, 2, 3];

        // iter() - borrows immutably, yields &T
        print!("iter() yields &T: ");
        for x in v.iter() {
            print!("{} ", x);  // x is &i32
            // *x = 10;  // error: cannot assign to immutable reference
        }
        println!();

        // iter_mut() - borrows mutably, yields &mut T
        print!("iter_mut() yields &mut T: ");
        for x in v.iter_mut() {
            *x += 10;  // can modify through mutable reference
            print!("{} ", x);
        }
        println!();

        // into_iter() - consumes collection, yields T
        print!("into_iter() yields T: ");
        for x in v.into_iter() {
            print!("{} ", x);  // x is i32, owns the value
        }
        println!();

        // v is no longer valid - ownership was transferred
        // println!("{:?}", v);  // error: borrow of moved value
    }

Iterator Adapters
-----------------

Iterator adapters transform iterators without consuming them. They're lazy - no
work happens until you call a consuming method like ``collect()``. This enables
efficient chaining of multiple operations. Each adapter below shows both C++ and
Rust approaches.

map - Transform Elements
~~~~~~~~~~~~~~~~~~~~~~~~

The ``map`` adapter applies a function to each element, producing a new iterator
of transformed values. This is equivalent to C++'s ``std::transform``:

**C++:**

.. code-block:: cpp

    #include <algorithm>
    #include <iostream>
    #include <vector>

    int main() {
      std::vector<int> v = {1, 2, 3};
      std::vector<int> doubled;

      // std::transform requires output iterator
      std::transform(v.begin(), v.end(), std::back_inserter(doubled),
                     [](int x) { return x * 2; });

      std::cout << "Doubled: ";
      for (int x : doubled) {
        std::cout << x << " ";  // 2 4 6
      }
      std::cout << "\n";

      return 0;
    }

**Rust:**

.. code-block:: rust

    fn main() {
        let v = vec![1, 2, 3];

        // map is lazy - nothing happens until collect()
        let doubled: Vec<i32> = v.iter().map(|x| x * 2).collect();
        println!("Doubled: {:?}", doubled);  // [2, 4, 6]

        // Can chain multiple maps
        let result: Vec<i32> = v.iter()
            .map(|x| x * 2)
            .map(|x| x + 1)
            .collect();
        println!("Doubled + 1: {:?}", result);  // [3, 5, 7]
    }

filter - Keep Matching Elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``filter`` adapter keeps only elements that satisfy a predicate. This is
equivalent to C++'s ``std::copy_if``:

**C++:**

.. code-block:: cpp

    #include <algorithm>
    #include <iostream>
    #include <vector>

    int main() {
      std::vector<int> v = {1, 2, 3, 4, 5};
      std::vector<int> evens;

      // copy_if is the filtering algorithm
      std::copy_if(v.begin(), v.end(), std::back_inserter(evens),
                   [](int x) { return x % 2 == 0; });

      std::cout << "Evens: ";
      for (int x : evens) {
        std::cout << x << " ";  // 2 4
      }
      std::cout << "\n";

      return 0;
    }

**Rust:**

.. code-block:: rust

    fn main() {
        let v = vec![1, 2, 3, 4, 5];

        // filter takes a predicate returning bool
        // Note: &&x because filter yields &T, and we're borrowing that
        let evens: Vec<i32> = v.iter()
            .filter(|&&x| x % 2 == 0)
            .cloned()  // convert &i32 to i32
            .collect();
        println!("Evens: {:?}", evens);  // [2, 4]

        // Alternative: use copied() instead of cloned() for Copy types
        let evens: Vec<i32> = v.iter()
            .filter(|&&x| x % 2 == 0)
            .copied()
            .collect();
        println!("Evens (copied): {:?}", evens);
    }

filter_map - Filter and Transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``filter_map`` adapter combines filtering and mapping in one step. It takes a
function returning ``Option<T>`` - ``Some`` values are kept, ``None`` values are
filtered out. This is particularly useful for parsing or fallible transformations:

**C++:**

.. code-block:: cpp

    #include <charconv>
    #include <iostream>
    #include <optional>
    #include <string>
    #include <vector>

    std::optional<int> try_parse(const std::string& s) {
      int value;
      auto result = std::from_chars(s.data(), s.data() + s.size(), value);
      if (result.ec == std::errc()) {
        return value;
      }
      return std::nullopt;
    }

    int main() {
      std::vector<std::string> strings = {"1", "two", "3", "four", "5"};
      std::vector<int> numbers;

      for (const auto& s : strings) {
        if (auto n = try_parse(s)) {
          numbers.push_back(*n);
        }
      }

      std::cout << "Parsed numbers: ";
      for (int x : numbers) {
        std::cout << x << " ";  // 1 3 5
      }
      std::cout << "\n";

      return 0;
    }

**Rust:**

.. code-block:: rust

    fn main() {
        let strings = vec!["1", "two", "3", "four", "5"];

        // filter_map: return Some to keep, None to filter out
        let numbers: Vec<i32> = strings
            .iter()
            .filter_map(|s| s.parse().ok())  // parse returns Result, .ok() converts to Option
            .collect();
        println!("Parsed numbers: {:?}", numbers);  // [1, 3, 5]

        // Equivalent using filter + map (less elegant)
        let numbers2: Vec<i32> = strings
            .iter()
            .filter(|s| s.parse::<i32>().is_ok())
            .map(|s| s.parse().unwrap())
            .collect();
        println!("Parsed (filter+map): {:?}", numbers2);
    }

take and skip
~~~~~~~~~~~~~

The ``take`` adapter limits iteration to the first N elements, while ``skip``
skips the first N elements. These are useful for pagination and windowing:

**C++:**

.. code-block:: cpp

    #include <iostream>
    #include <vector>

    int main() {
      std::vector<int> v = {1, 2, 3, 4, 5};

      // Take first 3 (manual loop or use ranges in C++20)
      std::cout << "First 3: ";
      for (size_t i = 0; i < 3 && i < v.size(); ++i) {
        std::cout << v[i] << " ";
      }
      std::cout << "\n";

      // Skip first 2
      std::cout << "Skip 2: ";
      for (size_t i = 2; i < v.size(); ++i) {
        std::cout << v[i] << " ";
      }
      std::cout << "\n";

      return 0;
    }

**Rust:**

.. code-block:: rust

    fn main() {
        let v = vec![1, 2, 3, 4, 5];

        // take: limit to first N elements
        let first_three: Vec<_> = v.iter().take(3).collect();
        println!("First 3: {:?}", first_three);  // [1, 2, 3]

        // skip: skip first N elements
        let skip_two: Vec<_> = v.iter().skip(2).collect();
        println!("Skip 2: {:?}", skip_two);  // [3, 4, 5]

        // Combine for pagination
        let page_size = 2;
        let page = 1;  // 0-indexed
        let page_items: Vec<_> = v.iter()
            .skip(page * page_size)
            .take(page_size)
            .collect();
        println!("Page 1 (size 2): {:?}", page_items);  // [3, 4]

        // take_while and skip_while use predicates
        let until_four: Vec<_> = v.iter().take_while(|&&x| x < 4).collect();
        println!("Take while < 4: {:?}", until_four);  // [1, 2, 3]
    }

enumerate - Add Indices
~~~~~~~~~~~~~~~~~~~~~~~

The ``enumerate`` adapter pairs each element with its index. This is cleaner than
maintaining a separate counter variable:

**C++:**

.. code-block:: cpp

    #include <iostream>
    #include <vector>
    #include <string>

    int main() {
      std::vector<std::string> v = {"apple", "banana", "cherry"};

      // Manual index tracking
      size_t i = 0;
      for (const auto& x : v) {
        std::cout << i << ": " << x << "\n";
        ++i;
      }

      // Or use traditional for loop
      for (size_t i = 0; i < v.size(); ++i) {
        std::cout << i << ": " << v[i] << "\n";
      }

      return 0;
    }

**Rust:**

.. code-block:: rust

    fn main() {
        let v = vec!["apple", "banana", "cherry"];

        // enumerate yields (index, element) tuples
        for (i, x) in v.iter().enumerate() {
            println!("{}: {}", i, x);
        }

        // Can destructure in closures too
        let indexed: Vec<_> = v.iter()
            .enumerate()
            .map(|(i, s)| format!("{}: {}", i, s))
            .collect();
        println!("Indexed: {:?}", indexed);
    }

zip - Combine Iterators
~~~~~~~~~~~~~~~~~~~~~~~

The ``zip`` adapter combines two iterators into one that yields pairs. Iteration
stops when either iterator is exhausted:

**C++:**

.. code-block:: cpp

    #include <iostream>
    #include <vector>
    #include <string>

    int main() {
      std::vector<int> numbers = {1, 2, 3};
      std::vector<std::string> words = {"one", "two", "three"};

      // Manual parallel iteration
      for (size_t i = 0; i < numbers.size() && i < words.size(); ++i) {
        std::cout << numbers[i] << " -> " << words[i] << "\n";
      }

      return 0;
    }

**Rust:**

.. code-block:: rust

    fn main() {
        let numbers = vec![1, 2, 3];
        let words = vec!["one", "two", "three"];

        // zip combines two iterators into pairs
        for (n, w) in numbers.iter().zip(words.iter()) {
            println!("{} -> {}", n, w);
        }

        // Collect into vector of tuples
        let pairs: Vec<_> = numbers.iter().zip(words.iter()).collect();
        println!("Pairs: {:?}", pairs);

        // Unequal lengths: stops at shorter
        let short = vec![1, 2];
        let long = vec!["a", "b", "c", "d"];
        let zipped: Vec<_> = short.iter().zip(long.iter()).collect();
        println!("Unequal zip: {:?}", zipped);  // [(1, "a"), (2, "b")]
    }

chain - Concatenate Iterators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``chain`` adapter concatenates two iterators, yielding all elements from the
first followed by all elements from the second:

**C++:**

.. code-block:: cpp

    #include <iostream>
    #include <vector>

    int main() {
      std::vector<int> a = {1, 2};
      std::vector<int> b = {3, 4};

      // Manual concatenation
      std::vector<int> combined;
      combined.insert(combined.end(), a.begin(), a.end());
      combined.insert(combined.end(), b.begin(), b.end());

      std::cout << "Combined: ";
      for (int x : combined) {
        std::cout << x << " ";
      }
      std::cout << "\n";

      return 0;
    }

**Rust:**

.. code-block:: rust

    fn main() {
        let a = vec![1, 2];
        let b = vec![3, 4];

        // chain concatenates iterators lazily
        let combined: Vec<_> = a.iter().chain(b.iter()).collect();
        println!("Combined: {:?}", combined);  // [1, 2, 3, 4]

        // Can chain multiple iterators
        let c = vec![5, 6];
        let all: Vec<_> = a.iter()
            .chain(b.iter())
            .chain(c.iter())
            .collect();
        println!("All: {:?}", all);  // [1, 2, 3, 4, 5, 6]
    }

flatten - Flatten Nested Iterators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``flatten`` adapter flattens nested iterators (or collections) into a single
iterator. This is useful for working with nested data structures:

**C++:**

.. code-block:: cpp

    #include <iostream>
    #include <vector>

    int main() {
      std::vector<std::vector<int>> nested = {{1, 2}, {3, 4}, {5}};

      // Manual flattening
      std::vector<int> flat;
      for (const auto& inner : nested) {
        for (int x : inner) {
          flat.push_back(x);
        }
      }

      std::cout << "Flattened: ";
      for (int x : flat) {
        std::cout << x << " ";
      }
      std::cout << "\n";

      return 0;
    }

**Rust:**

.. code-block:: rust

    fn main() {
        let nested = vec![vec![1, 2], vec![3, 4], vec![5]];

        // flatten collapses one level of nesting
        let flat: Vec<_> = nested.iter().flatten().collect();
        println!("Flattened: {:?}", flat);  // [1, 2, 3, 4, 5]

        // flat_map combines map + flatten (very common pattern)
        let words = vec!["hello", "world"];
        let chars: Vec<_> = words.iter()
            .flat_map(|s| s.chars())
            .collect();
        println!("All chars: {:?}", chars);  // ['h', 'e', 'l', 'l', 'o', 'w', 'o', 'r', 'l', 'd']
    }

Consuming Adapters
------------------

Consuming adapters (also called "terminal operations") consume the iterator and
produce a final result. Unlike adapters like ``map`` and ``filter``, these methods
trigger actual iteration.

collect - Gather into Collection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``collect`` method consumes an iterator and gathers elements into a collection.
The target type is inferred from context or specified with turbofish syntax:

**C++:**

.. code-block:: cpp

    #include <iostream>
    #include <set>
    #include <string>
    #include <vector>

    int main() {
      // Collecting into vector (manual)
      std::vector<int> v;
      for (int i = 1; i <= 5; ++i) {
        v.push_back(i);
      }

      // Collecting into set
      std::set<int> s(v.begin(), v.end());

      // Collecting chars into string
      std::vector<char> chars = {'h', 'e', 'l', 'l', 'o'};
      std::string str(chars.begin(), chars.end());

      std::cout << "String: " << str << "\n";

      return 0;
    }

**Rust:**

.. code-block:: rust

    use std::collections::HashSet;

    fn main() {
        // Collect range into Vec
        let v: Vec<i32> = (1..=5).collect();
        println!("Vec: {:?}", v);

        // Collect into HashSet (removes duplicates)
        let set: HashSet<i32> = vec![1, 2, 2, 3, 3, 3].into_iter().collect();
        println!("Set: {:?}", set);

        // Collect chars into String
        let s: String = ['h', 'e', 'l', 'l', 'o'].iter().collect();
        println!("String: {}", s);

        // Turbofish syntax when type can't be inferred
        let v = (1..=5).collect::<Vec<_>>();
        println!("Turbofish: {:?}", v);
    }

fold and reduce - Accumulate Values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``fold`` method accumulates values with an initial accumulator, while ``reduce``
uses the first element as the initial value. These are equivalent to C++'s
``std::accumulate`` and ``std::reduce``:

**C++:**

.. code-block:: cpp

    #include <iostream>
    #include <numeric>
    #include <vector>

    int main() {
      std::vector<int> v = {1, 2, 3, 4, 5};

      // accumulate with initial value (like fold)
      int sum = std::accumulate(v.begin(), v.end(), 0,
                                [](int acc, int x) { return acc + x; });
      std::cout << "Sum: " << sum << "\n";  // 15

      // reduce without initial value (C++17)
      // Note: std::reduce may reorder operations for parallelism
      int product = std::reduce(v.begin(), v.end(), 1, std::multiplies<int>());
      std::cout << "Product: " << product << "\n";  // 120

      return 0;
    }

**Rust:**

.. code-block:: rust

    fn main() {
        let v = vec![1, 2, 3, 4, 5];

        // fold: requires initial value, always succeeds
        let sum = v.iter().fold(0, |acc, x| acc + x);
        println!("Sum (fold): {}", sum);  // 15

        // reduce: uses first element as initial, returns Option
        let product = v.iter().copied().reduce(|acc, x| acc * x);
        println!("Product (reduce): {:?}", product);  // Some(120)

        // reduce on empty iterator returns None
        let empty: Vec<i32> = vec![];
        let result = empty.iter().copied().reduce(|acc, x| acc + x);
        println!("Empty reduce: {:?}", result);  // None

        // Building a string with fold
        let words = vec!["hello", "world"];
        let sentence = words.iter().fold(String::new(), |mut acc, &word| {
            if !acc.is_empty() {
                acc.push(' ');
            }
            acc.push_str(word);
            acc
        });
        println!("Sentence: {}", sentence);  // "hello world"
    }

sum and product
~~~~~~~~~~~~~~~

The ``sum`` and ``product`` methods are specialized folds for numeric types:

**C++:**

.. code-block:: cpp

    #include <iostream>
    #include <numeric>
    #include <vector>

    int main() {
      std::vector<int> v = {1, 2, 3, 4, 5};

      int sum = std::accumulate(v.begin(), v.end(), 0);
      int product = std::accumulate(v.begin(), v.end(), 1, std::multiplies<int>());

      std::cout << "Sum: " << sum << "\n";      // 15
      std::cout << "Product: " << product << "\n";  // 120

      return 0;
    }

**Rust:**

.. code-block:: rust

    fn main() {
        let v = vec![1, 2, 3, 4, 5];

        // sum and product require type annotation
        let sum: i32 = v.iter().sum();
        let product: i32 = v.iter().product();

        println!("Sum: {}", sum);      // 15
        println!("Product: {}", product);  // 120

        // Works with floating point too
        let floats = vec![1.5, 2.5, 3.0];
        let sum: f64 = floats.iter().sum();
        println!("Float sum: {}", sum);  // 7.0
    }

find and position
~~~~~~~~~~~~~~~~~

The ``find`` method returns the first element matching a predicate, while
``position`` returns its index:

**C++:**

.. code-block:: cpp

    #include <algorithm>
    #include <iostream>
    #include <vector>

    int main() {
      std::vector<int> v = {1, 2, 3, 4, 5};

      // find returns iterator
      auto it = std::find_if(v.begin(), v.end(), [](int x) { return x > 3; });
      if (it != v.end()) {
        std::cout << "Found: " << *it << "\n";  // 4
      }

      // position requires distance calculation
      auto pos = std::distance(v.begin(), it);
      std::cout << "Position: " << pos << "\n";  // 3

      return 0;
    }

**Rust:**

.. code-block:: rust

    fn main() {
        let v = vec![1, 2, 3, 4, 5];

        // find returns Option<&T>
        let found = v.iter().find(|&&x| x > 3);
        println!("Found: {:?}", found);  // Some(&4)

        // position returns Option<usize>
        let pos = v.iter().position(|&x| x > 3);
        println!("Position: {:?}", pos);  // Some(3)

        // Not found returns None
        let not_found = v.iter().find(|&&x| x > 10);
        println!("Not found: {:?}", not_found);  // None

        // find_map combines find and map
        let strings = vec!["1", "two", "3"];
        let first_num: Option<i32> = strings.iter().find_map(|s| s.parse().ok());
        println!("First parseable: {:?}", first_num);  // Some(1)
    }

any and all
~~~~~~~~~~~

The ``any`` method returns true if any element matches, while ``all`` returns true
if all elements match. These short-circuit on the first decisive result:

**C++:**

.. code-block:: cpp

    #include <algorithm>
    #include <iostream>
    #include <vector>

    int main() {
      std::vector<int> v = {1, 2, 3, 4, 5};

      bool has_even = std::any_of(v.begin(), v.end(), [](int x) { return x % 2 == 0; });
      bool all_positive = std::all_of(v.begin(), v.end(), [](int x) { return x > 0; });
      bool none_negative = std::none_of(v.begin(), v.end(), [](int x) { return x < 0; });

      std::cout << "Has even: " << (has_even ? "yes" : "no") << "\n";
      std::cout << "All positive: " << (all_positive ? "yes" : "no") << "\n";
      std::cout << "None negative: " << (none_negative ? "yes" : "no") << "\n";

      return 0;
    }

**Rust:**

.. code-block:: rust

    fn main() {
        let v = vec![1, 2, 3, 4, 5];

        // any: true if any element matches
        let has_even = v.iter().any(|&x| x % 2 == 0);
        println!("Has even: {}", has_even);  // true

        // all: true if all elements match
        let all_positive = v.iter().all(|&x| x > 0);
        println!("All positive: {}", all_positive);  // true

        // Short-circuit behavior
        let v2 = vec![1, 2, 3, 4, 5];
        let found_early = v2.iter().any(|&x| {
            println!("Checking {}", x);
            x == 2
        });
        // Only prints "Checking 1" and "Checking 2"
        println!("Found: {}", found_early);
    }

count, min, and max
~~~~~~~~~~~~~~~~~~~

These methods provide basic statistics about iterator elements:

**C++:**

.. code-block:: cpp

    #include <algorithm>
    #include <iostream>
    #include <vector>

    int main() {
      std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};

      size_t count = v.size();
      auto [min_it, max_it] = std::minmax_element(v.begin(), v.end());

      std::cout << "Count: " << count << "\n";
      std::cout << "Min: " << *min_it << "\n";
      std::cout << "Max: " << *max_it << "\n";

      return 0;
    }

**Rust:**

.. code-block:: rust

    fn main() {
        let v = vec![3, 1, 4, 1, 5, 9, 2, 6];

        // count consumes the iterator
        let count = v.iter().count();
        println!("Count: {}", count);  // 8

        // min and max return Option (None for empty iterators)
        let min = v.iter().min();
        let max = v.iter().max();
        println!("Min: {:?}", min);  // Some(&1)
        println!("Max: {:?}", max);  // Some(&9)

        // min_by and max_by for custom comparison
        let words = vec!["apple", "pie", "extraordinary"];
        let longest = words.iter().max_by_key(|s| s.len());
        println!("Longest: {:?}", longest);  // Some(&"extraordinary")
    }

C++ Comparison
--------------

This section provides a comprehensive comparison between C++ STL algorithms and
Rust iterator methods. C++ traditionally uses algorithm functions that take
iterator pairs, while Rust uses method chaining on iterator objects.

C++ STL Algorithms vs Rust Iterators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example shows a complete transformation pipeline in both languages.
Notice how Rust's method chaining is more concise and the lazy evaluation allows
the compiler to optimize the entire chain:

**C++ with algorithms:**

.. code-block:: cpp

    #include <algorithm>
    #include <iostream>
    #include <numeric>
    #include <vector>

    int main() {
      std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

      // Transform: double each element
      std::vector<int> doubled;
      std::transform(v.begin(), v.end(), std::back_inserter(doubled),
                     [](int x) { return x * 2; });

      // Filter: keep only evens (from original)
      std::vector<int> evens;
      std::copy_if(v.begin(), v.end(), std::back_inserter(evens),
                   [](int x) { return x % 2 == 0; });

      // Sum
      int sum = std::accumulate(v.begin(), v.end(), 0);

      // Chained operations require intermediate vectors
      std::vector<int> temp;
      std::copy_if(v.begin(), v.end(), std::back_inserter(temp),
                   [](int x) { return x % 2 == 0; });
      std::vector<int> result;
      std::transform(temp.begin(), temp.end(), std::back_inserter(result),
                     [](int x) { return x * 2; });

      std::cout << "Doubled: ";
      for (int x : doubled) std::cout << x << " ";
      std::cout << "\nEvens: ";
      for (int x : evens) std::cout << x << " ";
      std::cout << "\nSum: " << sum;
      std::cout << "\nFiltered+Doubled: ";
      for (int x : result) std::cout << x << " ";
      std::cout << "\n";

      return 0;
    }

**Rust equivalent:**

.. code-block:: rust

    fn main() {
        let v = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        // Transform: double each element
        let doubled: Vec<_> = v.iter().map(|x| x * 2).collect();

        // Filter: keep only evens
        let evens: Vec<_> = v.iter().filter(|&&x| x % 2 == 0).collect();

        // Sum
        let sum: i32 = v.iter().sum();

        // Chained operations - no intermediate allocations!
        let result: Vec<_> = v.iter()
            .filter(|&&x| x % 2 == 0)
            .map(|x| x * 2)
            .collect();

        println!("Doubled: {:?}", doubled);
        println!("Evens: {:?}", evens);
        println!("Sum: {}", sum);
        println!("Filtered+Doubled: {:?}", result);
    }

C++20 Ranges
~~~~~~~~~~~~

C++20 introduced ranges, which provide a more Rust-like experience with lazy
evaluation and method chaining via the pipe operator:

**C++20 Ranges:**

.. code-block:: cpp

    #include <iostream>
    #include <ranges>
    #include <vector>

    int main() {
      std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

      // Pipe syntax similar to Rust's method chaining
      auto result = v
          | std::views::filter([](int x) { return x % 2 == 0; })
          | std::views::transform([](int x) { return x * 2; });

      // Lazy evaluation - nothing computed until iteration
      std::cout << "Filtered and doubled evens: ";
      for (int x : result) {
        std::cout << x << " ";  // 4 8 12 16 20
      }
      std::cout << "\n";

      // Take first N elements
      auto first_three = v | std::views::take(3);
      std::cout << "First three: ";
      for (int x : first_three) {
        std::cout << x << " ";
      }
      std::cout << "\n";

      return 0;
    }

**Rust equivalent:**

.. code-block:: rust

    fn main() {
        let v = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        // Method chaining (Rust's native style)
        let result: Vec<_> = v.iter()
            .filter(|&&x| x % 2 == 0)
            .map(|x| x * 2)
            .collect();

        println!("Filtered and doubled evens: {:?}", result);  // [4, 8, 12, 16, 20]

        // Take first N elements
        let first_three: Vec<_> = v.iter().take(3).collect();
        println!("First three: {:?}", first_three);
    }

Creating Custom Iterators
-------------------------

You can create custom iterators by implementing the ``Iterator`` trait. This
requires defining the ``Item`` associated type and the ``next`` method. Once
implemented, your type automatically gains access to all iterator adapter methods.

The following example creates a counter iterator that yields numbers from 1 to a
maximum value:

**C++:**

.. code-block:: cpp

    #include <iostream>
    #include <iterator>

    class Counter {
    public:
      using iterator_category = std::input_iterator_tag;
      using value_type = int;
      using difference_type = std::ptrdiff_t;
      using pointer = int*;
      using reference = int&;

    private:
      int count_;
      int max_;

    public:
      Counter(int max) : count_(0), max_(max) {}

      // For end sentinel
      static Counter end() {
        Counter c(0);
        c.count_ = c.max_ + 1;
        return c;
      }

      int operator*() const { return count_; }

      Counter& operator++() {
        ++count_;
        return *this;
      }

      bool operator!=(const Counter& other) const {
        return count_ <= max_;
      }
    };

    int main() {
      // Manual iteration
      int sum = 0;
      for (Counter c(5); c != Counter::end(); ++c) {
        if (*c > 0) {  // skip 0
          sum += *c;
        }
      }
      std::cout << "Sum 1-5: " << sum << "\n";  // 15

      return 0;
    }

**Rust:**

.. code-block:: rust

    struct Counter {
        count: u32,
        max: u32,
    }

    impl Counter {
        fn new(max: u32) -> Self {
            Counter { count: 0, max }
        }
    }

    impl Iterator for Counter {
        type Item = u32;

        fn next(&mut self) -> Option<Self::Item> {
            if self.count < self.max {
                self.count += 1;
                Some(self.count)
            } else {
                None
            }
        }
    }

    fn main() {
        // All iterator methods are automatically available
        let sum: u32 = Counter::new(5).sum();
        println!("Sum 1-5: {}", sum);  // 15

        // Can use any adapter
        let evens: Vec<_> = Counter::new(10)
            .filter(|&x| x % 2 == 0)
            .collect();
        println!("Evens 1-10: {:?}", evens);  // [2, 4, 6, 8, 10]

        // Chaining with other iterators
        let doubled: Vec<_> = Counter::new(3)
            .map(|x| x * 2)
            .collect();
        println!("Doubled 1-3: {:?}", doubled);  // [2, 4, 6]
    }

See Also
--------

- :doc:`rust_container` - Collections that implement Iterator
- :doc:`rust_closure` - Closures used with iterator adapters
