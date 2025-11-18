======
Ranges
======

.. contents:: Table of Contents
    :backlinks: none

range-v3 - debug a vector
-------------------------

.. code-block:: cpp

    // g++ -O3 -std=c++17 -Wall -Werror -I${HDR} a.cpp

    #include <iostream>
    #include <vector>
    #include <range/v3/view/all.hpp>

    int main(int argc, char *argv[]) {
      std::vector<int> v{5, 4, 3, 2, 1, 1, 1};
      std::cout << ranges::views::all(v) << "\n";
      // [5,4,3,2,1,1,1]
    }

range-v3 - all_of
-----------------

.. code-block:: cpp

    #include <iostream>
    #include <vector>
    #include <range/v3/algorithm/all_of.hpp>

    int main(int argc, char *argv[]) {
      std::vector<int> v{0, 2, 4};
      std::cout << ranges::all_of(v, [](auto &&x) { return !(x % 2); });
      // 1
    }

.. code-block:: python

    >>> a = [0, 2, 4]
    >>> all(x % 2 == 0 for x in a)
    True

range-v3 - any_of
-----------------

.. code-block:: cpp

    #include <iostream>
    #include <vector>
    #include <range/v3/algorithm/any_of.hpp>

    int main(int argc, char *argv[]) {
      std::vector<int> v{0, 1, 2};
      std::cout << ranges::any_of(v, [](auto &&x) { return !(x % 2); });
      // 1
    }

.. code-block:: python

    >>> a = [0, 1 ,2]
    >>> any(x % 2 == 0 for x in a)
    True

range-v3 - slice
----------------

.. code-block:: cpp

    #include <iostream>
    #include <vector>
    #include <range/v3/algorithm/copy.hpp>
    #include <range/v3/action/slice.hpp>
    #include <range/v3/view/slice.hpp>
    #include <range/v3/view/all.hpp>

    int main(int argc, char *argv[]) {
      std::vector<int> x{5, 4, 3, 2, 1};

      auto y = x | ranges::copy | ranges::actions::slice(1, 3);
      std::cout << ranges::views::all(y) << "\n";
      // [4,3]

      for (auto &&e : x | ranges::views::slice(2, 4)) {
        std::cout << e << "\n";
      }
    }

.. code-block:: python

    >>> a = [5,4,3,2,1]
    >>> print(a[1:3])
    [4, 3]

range-v3 - enumerate
--------------------

.. code-block:: cpp

    #include <iostream>
    #include <vector>
    #include <range/v3/view/enumerate.hpp>

    int main(int argc, char *argv[]) {
      std::vector<int> v{5, 4, 3, 2, 1, 1, 1};
      for (auto &&[i, e] : v |  ranges::views::enumerate) {
        std::cout << i << ", " << e << "\n";
      }
    }

.. code-block:: python

    >>> a = [5,4,3,2,1,1]
    >>> for i, e in enumerate(a):
    ...     print(i, e)
    ...
    0 5
    1 4
    2 3
    3 2
    4 1
    5 1

range-v3 - concat vectors
-------------------------

.. code-block:: cpp

    #include <iostream>
    #include <vector>
    #include <range/v3/view/concat.hpp>
    #include <range/v3/view/all.hpp>

    int main(int argc, char *argv[]) {
      std::vector<int> x{1, 5};
      std::vector<int> y{2, 8};
      std::vector<int> z{0, 3};
      auto r = ranges::views::concat(x, y, z);
      std::cout << ranges::views::all(r) << "\n";
      // [1,5,2,8,0,3]
    }

.. code-block:: python

    >>> a = [1, 5]
    >>> b = [2, 8]
    >>> c = [0, 3]
    >>> print(a + b + c)
    [1, 5, 2, 8, 0, 3]

range-v3 - accumulate (sum)
---------------------------

.. code-block:: cpp

    #include <iostream>
    #include <vector>
    #include <range/v3/numeric/accumulate.hpp>

    int main(int argc, char *argv[]) {
      std::vector<int> v{1, 2, 3, 4, 5};
      const auto r = ranges::accumulate(v, 0);
      std::cout << r << "\n";
      // 15
    }

.. code-block:: python

    >>> a = [1, 2, 3, 4, 5]
    >>> sum(a)
    15

range-v3 - accumulate (reduce)
------------------------------

.. code-block:: cpp

    #include <iostream>
    #include <vector>
    #include <range/v3/numeric/accumulate.hpp>
    #include <range/v3/view/all.hpp>

    int main(int argc, char *argv[]) {
      std::vector<int> v{1, 2, 3, 4, 5};
      const auto r = ranges::accumulate(v, 1, [](auto &a, auto &b){
        return a + b;
      });
      std::cout << r << "\n";
      // 120
    }

.. code-block:: python

    >>> from functools import reduce
    >>> reduce(lambda x, y: x * y, [1, 2, 3, 4, 5], 1)
    120

range-v3 - sort
---------------

.. code-block:: cpp

    #include <iostream>
    #include <vector>
    #include <range/v3/action/sort.hpp>
    #include <range/v3/view/all.hpp>

    int main(int argc, char *argv[]) {
      std::vector<int> v{5, 4, 3, 2, 1, 1, 1};
      v |= ranges::actions::sort;
      std::cout << ranges::views::all(v) << "\n";
      // [1,1,1,2,3,4,5]
    }

.. code-block:: python

    >>> a = [5,4,3,2,1,1,1]
    >>> a.sort()
    >>> a
    [1, 1, 1, 2, 3, 4, 5]

range-v3 - reverse sort
-----------------------

.. code-block:: cpp

    #include <iostream>
    #include <vector>
    #include <range/v3/action/sort.hpp>
    #include <range/v3/action/reverse.hpp>
    #include <range/v3/view/all.hpp>

    int main(int argc, char *argv[]) {
      std::vector<int> v{1, 5, 3, 2, 6};
      v |= ranges::actions::sort | ranges::actions::reverse;
      std::cout << ranges::views::all(v) << "\n";
    }

.. code-block:: python

    >>> a = [1, 5, 3, 2, 6]
    >>> a.sort(reverse=True)
    >>> a
    [6, 5, 3, 2, 1]

range-v3 - sort & uniqe
-----------------------

.. code-block:: cpp

    // echo 5 4 3 2 1 1 1 | tr -s " " "\n" | sort | uniq

    #include <iostream>
    #include <vector>
    #include <range/v3/action/unique.hpp>
    #include <range/v3/action/sort.hpp>
    #include <range/v3/view/all.hpp>

    int main(int argc, char *argv[]) {
      std::vector<int> v{5, 4, 3, 2, 1, 1, 1};
      v |= ranges::actions::sort | ranges::actions::unique;
      std::cout << ranges::views::all(v) << "\n";
      // [1,2,3,4,5]
    }

.. code-block:: python

    >>> a = [5, 4, 3, 2, 1, 1, 1]
    >>> a = list({x for x in a})
    >>> a.sort()
    >>> a
    [1, 2, 3, 4, 5]

range-v3 - zip
--------------

.. code-block:: cpp

    #include <iostream>
    #include <vector>
    #include <range/v3/view/zip.hpp>
    #include <range/v3/view/all.hpp>

    int main(int argc, char *argv[]) {
      std::vector<int> x{5, 4, 3, 2};
      std::vector<int> y{1, 2, 3 ,4};

      for (auto &&[a, b] : ranges::views::zip(x, y)) {
        std::cout << a << " " << b << "\n";
      }
    }

.. code-block:: python

    >>> a = [5,4,3,2]
    >>> b = [1,2,3,4]
    >>> for x, y in zip(a, b):
    ...     print(x, y)
    ...
    5 1
    4 2
    3 3
    2 4

range-v3 - split
----------------

.. code-block:: cpp

    #include <iostream>
    #include <vector>
    #include <string>
    #include <range/v3/view/c_str.hpp>
    #include <range/v3/action/split.hpp>
    #include <range/v3/view/all.hpp>

    int main(int argc, char *argv[]) {
      std::string s = "hello c++";
      auto v = ranges::actions::split(s, ranges::views::c_str(" "));
      std::cout << ranges::views::all(v) << "\n";
      // [hello,c++]
    }

.. code-block:: python

    >>> s = "hello python"
    >>> s.split(" ")
    ['hello', 'python']

range-v3 - tokenize
-------------------

.. code-block:: cpp

    #include <iostream>
    #include <vector>
    #include <string>
    #include <regex>
    #include <range/v3/view/tokenize.hpp>
    #include <range/v3/view/all.hpp>

    int main(int argc, char *argv[]) {
      const std::string s = "hello cpp";
      const auto p = std::regex{"[\\w]+"};
      auto r = s | ranges::views::tokenize(p);
      std::cout << ranges::views::all(r) << "\n";
    }

.. code-block:: python

    >>> import re
    >>> s = "hello python"
    >>> [m.group() for m in re.finditer(r"\w+", s)]
    ['hello', 'python']

range-v3 - join
---------------

.. code-block:: cpp

    #include <iostream>
    #include <vector>
    #include <string>
    #include <range/v3/core.hpp>
    #include <range/v3/view/join.hpp>
    #include <range/v3/view/all.hpp>

    int main(int argc, char *argv[]) {
      std::vector<std::string> v{"hello", "c++"};
      auto s = v | ranges::views::join(' ') | ranges::to<std::string>();
      std::cout << s << "\n";
    }

.. code-block:: python

    >>> v = ['hello', 'python']
    >>> ' '.join(v)
    'hello python'

range-v3 - split + join
-----------------------

.. code-block:: cpp

    #include <iostream>
    #include <vector>
    #include <string>
    #include <range/v3/view/c_str.hpp>
    #include <range/v3/action/split.hpp>
    #include <range/v3/view/join.hpp>

    int main(int argc, char *argv[]) {
      std::string s = "a b c d e";
      auto v = ranges::actions::split(s, ranges::views::c_str(" "));
      auto x = v | ranges::views::join(',') | ranges::to<std::string>();
      std::cout << x << "\n";
    }

.. code-block:: python

    >>> s = "a b c d e"
    >>> ",".join(s.split(" "))
    'a,b,c,d,e'

range-v3 - iota
---------------

.. code-block:: cpp

    #include <iostream>
    #include <range/v3/view/iota.hpp>
    #include <range/v3/view/all.hpp>

    int main(int argc, char *argv[]) {
      auto seq = ranges::views::iota(5, 8);
      std::cout << ranges::views::all(seq) << "\n";
      // [5,6,7]
    }

.. code-block:: python

    >>> [x for x in range(5, 8)]
    [5, 6, 7]

range-v3 - generate
-------------------

.. code-block:: cpp

    #include <iostream>
    #include <vector>
    #include <range/v3/view/generate.hpp>
    #include <range/v3/view/take.hpp>
    #include <range/v3/view/all.hpp>

    int main(int argc, char *argv[]) {
      auto fib = ranges::views::generate([i=0, j=1]() mutable {
        int tmp = i; i+= j; j = i; return tmp;
      });

      auto v = fib | ranges::views::take(5);
      std::cout << ranges::views::all(v) << std::endl;
      // [0,1,2,4,8]
    }

.. code-block:: python

    >>> def fib(n):
    ...     a, b = 0, 1
    ...     for _ in range(n):
    ...         yield a
    ...         a, b = b, a + b
    ...
    >>> [x for x in fib(5)]
    [0, 1, 1, 2, 3]

range-v3 - take
---------------

.. code-block:: cpp

    #include <iostream>
    #include <range/v3/view/iota.hpp>
    #include <range/v3/view/take.hpp>
    #include <range/v3/view/all.hpp>

    int main(int argc, char *argv[]) {
      auto v = ranges::views::iota(5, 10) | ranges::views::take(3);
      std::cout << ranges::views::all(v) << "\n";
      // [5,6,7]
    }

range-v3 - take_while
---------------------

.. code-block:: cpp

    #include <iostream>
    #include <range/v3/view/iota.hpp>
    #include <range/v3/view/take_while.hpp>
    #include <range/v3/view/all.hpp>

    int main(int argc, char *argv[]) {
      auto v = ranges::views::iota(5, 10)
          | ranges::views::take_while([](auto &&x) { return x < 8; });
      std::cout << ranges::views::all(v) << "\n";
    }


range-v3 - drop
---------------

.. code-block:: cpp

    #include <iostream>
    #include <vector>
    #include <range/v3/action/drop.hpp>
    #include <range/v3/view/all.hpp>

    int main(int argc, char *argv[]) {
      std::vector<int> v{1, 2, 3, 4, 5, 6};
      v |= ranges::actions::drop(3);
      std::cout << ranges::views::all(v) << "\n";
    }

range-v3 - drop_while
---------------------

.. code-block:: cpp

    #include <iostream>
    #include <range/v3/view/iota.hpp>
    #include <range/v3/view/drop_while.hpp>
    #include <range/v3/view/all.hpp>

    int main(int argc, char *argv[]) {
      auto v = ranges::views::iota(5, 10)
          | ranges::views::drop_while([](auto &&x) { return x < 8; });
      std::cout << ranges::views::all(v) << "\n";
    }

range-v3 - transform (map)
--------------------------

.. code-block:: cpp

    #include <iostream>
    #include <vector>
    #include <range/v3/view/transform.hpp>
    #include <range/v3/view/all.hpp>

    int main(int argc, char *argv[]) {
      std::vector<int> v{1, 2, 3, 4, 5};
      auto r = v | ranges::views::transform([](auto &&x){ return x*x; });
      std::cout << ranges::views::all(r) << "\n";
      // [1,4,9,16,25]
    }

range-v3 - filter
-----------------

.. code-block:: cpp

    #include <iostream>
    #include <vector>
    #include <range/v3/view/filter.hpp>
    #include <range/v3/view/all.hpp>

    int main(int argc, char *argv[]) {
      std::vector<int> v{1, 2, 3, 4, 5};
      auto r = v | ranges::views::filter([](auto &&x){ return x > 3; });
      std::cout << ranges::views::all(r) << "\n";
      // [4,5]
    }

range-v3 - group_by
-------------------

.. code-block:: cpp

    #include <iostream>
    #include <string>
    #include <range/v3/view/group_by.hpp>
    #include <range/v3/view/all.hpp>

    int main(int argc, char *argv[]) {
      std::string s = "aaaabbbccd";
      auto r = s | ranges::views::group_by([](auto &&x, auto &&y){
        return x == y;
      });
      std::cout << ranges::views::all(r) << "\n";
      // [[a,a,a,a],[b,b,b],[c,c],[d]]
    }

range-v3 - cycle
----------------

.. code-block:: cpp

    #include <iostream>
    #include <vector>
    #include <range/v3/view/cycle.hpp>
    #include <range/v3/view/take.hpp>
    #include <range/v3/view/all.hpp>

    int main(int argc, char *argv[]) {
      std::vector<int> v{1, 2, 3};
      auto r = v | ranges::views::cycle | ranges::views::take(6);
      std::cout << ranges::views::all(r) << "\n";
    }

range-v3 - keys
---------------

.. code-block:: cpp

    #include <iostream>
    #include <unordered_map>
    #include <range/v3/view/map.hpp>
    #include <range/v3/view/all.hpp>

    int main(int argc, char *argv[]) {
      std::unordered_map<int, int> m{{9, 5}, {2, 7}};
      auto keys = m | ranges::views::keys;
      for (auto &&k : keys) {
        std::cout << k << "\n";
      }
    }

range-v3 - values
-----------------

.. code-block:: cpp

    #include <iostream>
    #include <unordered_map>
    #include <range/v3/view/map.hpp>
    #include <range/v3/view/all.hpp>

    int main(int argc, char *argv[]) {
      std::unordered_map<int, int> m{{9, 5}, {2, 7}};
      auto values = m | ranges::views::values;
      for (auto &&v : values) {
        std::cout << v << "\n";
      }
    }

range-v3 - cartesian_product
----------------------------

.. code-block:: cpp

    #include <iostream>
    #include <vector>
    #include <string>
    #include <range/v3/view/cartesian_product.hpp>

    int main(int argc, char *argv[]) {
      std::string x = "ab";
      std::vector<int> y{1, 2};
      auto r = ranges::views::cartesian_product(x, y);
      for (auto &&[a, b] : r) {
        std::cout << a << b << "\n";
      }
      // a1 a2 b1 b2
    }

range-v3 - permutation
----------------------

.. code-block:: cpp

    #include <iostream>
    #include <vector>
    #include <range/v3/algorithm/permutation.hpp>
    #include <range/v3/view/all.hpp>

    int main(int argc, char *argv[]) {
      std::vector<int> v{1, 2, 3};
      do {
        std::cout << ranges::views::all(v) << "\n";
      } while (ranges::next_permutation(v));
    }

c++20 range - iota
------------------

.. code-block:: cpp

    // g++-10 -Wall -Werror -O3 -g --std=c++20 a.cc

    #include <iostream>
    #include <ranges>

    int main(int argc, char *argv[])
    {
      using namespace std::ranges;

      for (auto i : views::iota(1) | views::take(5)) {
        std::cout << i << std::endl;
      }
    }

c++20 range - transform
-----------------------

.. code-block:: cpp

    #include <iostream>
    #include <ranges>
    #include <vector>

    int main(int argc, char *argv[])
    {
      using namespace std::ranges;

      std::vector v{1, 2, 3};
      auto adaptor = views::transform([](auto &e) { return e * e; });
      for (auto i : v | adaptor) {
        std::cout << i << std::endl;
      }
    }

c++20 range - filter
--------------------

.. code-block:: cpp

    #include <iostream>
    #include <ranges>
    #include <vector>

    int main(int argc, char *argv[])
    {
      using namespace std::ranges;

      std::vector v{1, 2, 3};
      auto adaptor = views::filter([](auto &e) { return e % 2 == 0; });

      for (auto i : v | adaptor) {
        std::cout << i << std::endl;
      }
    }


c++20 range - split
-------------------

.. code-block:: cpp

    #include <iostream>
    #include <ranges>
    #include <string>

    int main(int argc, char *argv[])
    {
      using namespace std::ranges;
      std::string s{"This is a string."};

      for (auto v : s | views::split(' ')) {
        std::string w;
        for (auto &c : v) {
          w += c;
        }
        std::cout << w << std::endl;
      }
    }

c++20 range - join
------------------

.. code-block:: cpp

    #include <iostream>
    #include <ranges>
    #include <vector>
    #include <string>

    int main(int argc, char *argv[])
    {
      using namespace std::ranges;
      std::vector<std::string> v{"This", " ", "is", " ", "a", " ", "string."};
      std::string s;
      for (auto &c : v | views::join) {
        s += c;
      }
      std::cout << s << std::endl;
    }
