========
Iterator
========

.. contents:: Table of Contents
    :backlinks: none

Upper Bound
-----------

Note that `std::upper_bound(x.begin(), x.end(), val)` finds an element which
is *greater than the val*. However, `std::lower_bound(x.begin(), x.end(), val)`,
finds an element which is *greater or equal to the val*.

.. code-block:: cpp

    #include <iostream>
    #include <deque>
    #include <algorithm>

    int main(int argc, char *argv[]) {
      std::deque<int> v{1,2,3,4,5,7,10};

      auto x = 5;
      auto pos1 = std::upper_bound(v.begin(), v.end(), x);
      std::cout << *pos1 << "\n";

      auto pos2 = std::lower_bound(v.begin(), v.end(), x);
      std::cout << *pos2 << "\n";

      // 7
      // 5
    }

Insert an Element into a Sorted List
------------------------------------

.. code-block:: cpp

    #include <iostream>
    #include <deque>
    #include <algorithm>
    #include <range/v3/view/all.hpp>


    int main(int argc, char *argv[]) {
      std::deque<int> v{1,2,3,4,5,7,10};

      auto x = 8;
      auto pos = std::upper_bound(v.begin(), v.end(), x);
      v.insert(pos, x);
      std::cout << ranges::views::all(v) << "\n";
      // [1,2,3,4,5,7,8,10]
    }

Erase an Element in a Sorted List
---------------------------------

.. code-block:: cpp

    #include <iostream>
    #include <deque>
    #include <algorithm>
    #include <range/v3/view/all.hpp>


    int main(int argc, char *argv[]) {
      std::deque<int> v{1,2,3,4,5,7,10};

      auto x = 7;
      auto pos = std::lower_bound(v.begin(), v.end(), x);
      v.erase(pos);
      std::cout << ranges::views::all(v) << "\n";
      // [1,2,3,4,5,10]
    }

Reverse Range-based for Loop
----------------------------

.. code-block:: cpp

    // via boost
    // $ g++ --std=c++14 -Wall -Werror -g -O3 reverse.cpp
    // $ ./a.out
    // dlrow olleh

    #include <iostream>
    #include <string>
    #include <boost/range/adaptor/reversed.hpp>

    using namespace boost;

    int main(int argc, char *argv[]) {
      std::string in = "hello world";
      std::string out;
      for (const auto &c : adaptors::reverse(in)) {
          out += c;
      }
      std::cout << out << "\n";
    }

Customize an Iterator
---------------------

.. code-block:: cpp

    // $ g++ -std=c++17 -Wall -Werror -g -O3 a.cc

    #include <iostream>
    #include <memory>

    template <typename T>
    class Array
    {
     public:
      class iterator
      {
        public:
          iterator(T *ptr) : ptr_{ptr} {}
          iterator operator++() { auto i = *this; ++ptr_; return i; }
          iterator operator++(int) { ++ptr_; return *this;};
          T &operator*() { return *ptr_; }
          T *operator->() { return ptr_; }
          bool operator==(const iterator &rhs) { return ptr_ == rhs.ptr_; }
          bool operator!=(const iterator &rhs) { return ptr_ != rhs.ptr_; }
        private:
          T *ptr_;
      };

      class const_iterator
      {
        public:
         const_iterator(T *ptr) : ptr_{ptr} {}
         const_iterator operator++() { auto i = *this; ++ptr_; return i; }
         const_iterator operator++(int) { ++ptr_; return *this; }
         const T &operator*() const { return *ptr_; }
         const T *operator->() const { return ptr_; }
         bool operator==(const const_iterator &rhs) { return ptr_ == rhs.ptr_; }
         bool operator!=(const const_iterator &rhs) { return ptr_ != rhs.ptr_; }
        private:
         T *ptr_;
      };

      Array(size_t size) : size_(size), data_{std::make_unique<T[]>(size)} {}
      size_t size() const { return size_; }
      T &operator[](size_t i) { return data_[i]; };
      const T &operator[](size_t i) const { return data_[i]; }
      iterator begin() { return iterator(data_.get()); }
      iterator end() { return iterator(data_.get() + size_); }
      const_iterator cbegin() const { return const_iterator(data_.get()); }
      const_iterator cend() const { return const_iterator(data_.get() + size_); }

     private:
      size_t size_;
      std::unique_ptr<T[]> data_;
    };



    int main(int argc, char *argv[])
    {
      Array<double> points(2);
      points[0] = 55.66;
      points[1] = 95.27;
      for (auto &e : points) {
        std::cout << e << "\n";
      }
      for (auto it = points.cbegin(); it != points.cend(); ++it) {
        std::cout << *it << "\n";
      }
    }

Iterate an Internal Vector
--------------------------

.. code-block:: cpp

    #include <iostream>
    #include <utility>
    #include <vector>

    template<typename T>
    class Vector {
     public:
      using iterator = typename std::vector<T>::iterator;
      using const_iterator = typename std::vector<T>::const_iterator;

      inline iterator begin() noexcept {return v.begin();}
      inline iterator end() noexcept {return v.end();}
      inline const_iterator cbegin() const noexcept {return v.cbegin();}
      inline const_iterator cend() const noexcept {return v.cend();}

      template<class... Args>
      auto emplace_back(Args&&... args) {
          return v.emplace_back(std::forward<Args>(args)...);
      }
     private:
      std::vector<T> v;
    };


    int main(int argc, char *argv[]) {
      Vector<int> v;
      v.emplace_back(1);
      v.emplace_back(2);
      v.emplace_back(3);

      for (auto &it : v) {
          std::cout << it << std::endl;
      }
      return 0;
    }

Iterate a file
--------------

.. code-block:: cpp

    // $ g++ -std=c++17 -Wall -Werror -g -O3 a.cc
    // $ ./a.out file

    #include <iostream>
    #include <iterator>
    #include <fstream>
    #include <string>

    class line : public std::string {};

    std::istream &operator>>(std::istream &is, line &l)
    {
      std::getline(is, l);
      return is;
    }

    class FileReader
    {
     public:
      using iterator = std::istream_iterator<line>;
      inline iterator begin() noexcept { return begin_; }
      inline iterator end() noexcept { return end_; }

     public:
      FileReader(const std::string path) : f_{path}, begin_{f_} {}
      friend std::istream &operator>>(std::istream &, std::string &);

     private:
      std::ifstream f_;
      iterator begin_;
      iterator end_;
    };

    int main(int argc, char *argv[])
    {
      FileReader reader(argv[1]);
      for (auto &line : reader) {
        std::cout << line << "\n";
      }
    }

Position after Erasing
----------------------

.. code-block:: cpp

    // deque
	#include <iostream>
	#include <deque>
	#include <range/v3/view/all.hpp>

	int main(int argc, char *argv[]) {
	  std::deque<int> q{1, 2, 3, 4, 5};
	  auto it = q.begin() + 2;

	  std::cout << *it << "\n";
	  std::cout << ranges::views::all(q) << "\n";

	  q.erase(it);
	  std::cout << *it << "\n";
	  std::cout << ranges::views::all(q) << "\n";

	  // output
	  //   3
	  //   [1,2,3,4,5]
	  //   4
	  //   [1,2,4,5]
	}

.. code-block:: cpp

	#include <iostream>
	#include <vector>
	#include <range/v3/view/all.hpp>

	int main(int argc, char *argv[]) {
	  std::vector<int> v{1, 2, 3, 4, 5};
	  auto it = v.begin() + 2;

	  std::cout << *it << "\n";
	  std::cout << ranges::views::all(v) << "\n";

	  v.erase(it);
	  std::cout << *it << "\n";
	  std::cout << ranges::views::all(v) << "\n";

	  // output
	  //   3
	  //   [1,2,3,4,5]
	  //   4
	  //   [1,2,4,5]
	}


.. code-block:: cpp

	#include <iostream>
	#include <list>
	#include <range/v3/view/all.hpp>

	int main(int argc, char *argv[]) {
	  std::list<int> l{1, 2, 3, 4, 5};
	  auto it = l.begin();
	  ++it;

	  std::cout << *it << "\n";
	  std::cout << ranges::views::all(l) << "\n";

	  // Note that Iterators, pointers and references referring to elements
	  // removed by the function are invalidated. This is an example to show
	  // that an iterator do not point to the next element after erasing.
	  l.erase(it);
	  std::cout << *it << "\n";
	  std::cout << ranges::views::all(l) << "\n";
	  // output
	  //   2
	  //   [1,2,3,4,5]
	  //   2
	  //   [1,3,4,5]
	}


Vector Comparision
------------------

Note that the comparision operators are removed in C++20 (see [doc](https://en.cppreference.com/w/cpp/container/vector)).
Therefore, using a lambda function as compare function is better than using
default comparision when elements are not builtin types or has its own comparision
operators.

.. code-block:: cpp

    #include <iostream>
    #include <vector>

    int main(int argc, char *argv[]) {
        std::vector<int> v1{5,2};
        std::vector<int> v2{2,3,4};
        std::cout << (v1 < v2) << "\n";
        // output: 0
    }
