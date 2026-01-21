======
Thrust
======

.. meta::
   :description: Thrust library tutorial covering STL-like GPU algorithms including vector containers, transforms, reductions, sorting, and custom functors for CUDA parallel programming.
   :keywords: Thrust, CUDA, GPU algorithms, parallel STL, device_vector, thrust::transform, thrust::reduce, thrust::sort, GPU programming, NVIDIA

.. contents:: Table of Contents
    :backlinks: none

Thrust is a parallel algorithms library that provides an STL-like interface for
GPU programming. It abstracts away CUDA kernel launches and memory management,
allowing developers to write high-level code that automatically executes on the
GPU. Thrust supports multiple backends including CUDA, OpenMP, and TBB, making
code portable across different parallel architectures. The library includes
containers (``device_vector``, ``host_vector``), iterators, and a comprehensive
set of parallel algorithms for transformations, reductions, sorting, and scans.

Device and Host Vectors
-----------------------

:Source: `src/cuda/thrust-basics <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/thrust-basics>`_

Thrust provides ``device_vector`` for GPU memory and ``host_vector`` for CPU memory.
Data transfers between host and device happen automatically through assignment,
making memory management straightforward. Both containers behave like ``std::vector``
with familiar operations like ``push_back``, ``resize``, and iterator access.

.. code-block:: cpp

    #include <thrust/device_vector.h>
    #include <thrust/host_vector.h>

    int main() {
      // Create host vector and initialize
      thrust::host_vector<int> h_vec(4);
      h_vec[0] = 10;
      h_vec[1] = 20;
      h_vec[2] = 30;
      h_vec[3] = 40;

      // Transfer to device (automatic copy)
      thrust::device_vector<int> d_vec = h_vec;

      // Modify on device
      d_vec[0] = 100;  // Note: slow, triggers kernel launch

      // Transfer back to host
      thrust::host_vector<int> result = d_vec;
    }

Transform
---------

:Source: `src/cuda/thrust-transform <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/thrust-transform>`_

``thrust::transform`` applies a function to each element, similar to ``std::transform``.
It supports unary operations on a single range and binary operations combining two
ranges. Thrust provides built-in functors like ``thrust::negate`` and ``thrust::plus``,
or you can use lambdas with ``__host__ __device__`` annotations.

.. code-block:: cpp

    #include <thrust/device_vector.h>
    #include <thrust/functional.h>
    #include <thrust/transform.h>

    int main() {
      thrust::device_vector<int> a(4, 10);  // [10, 10, 10, 10]
      thrust::device_vector<int> b(4, 5);   // [5, 5, 5, 5]
      thrust::device_vector<int> c(4);

      // Unary transform: negate each element
      thrust::transform(a.begin(), a.end(), c.begin(), thrust::negate<int>());
      // c = [-10, -10, -10, -10]

      // Binary transform: add two vectors
      thrust::transform(a.begin(), a.end(), b.begin(), c.begin(), thrust::plus<int>());
      // c = [15, 15, 15, 15]

      // Custom functor with lambda (CUDA 11+)
      thrust::transform(
          a.begin(),
          a.end(),
          c.begin(),
          [] __host__ __device__(int x) { return x * x; }
      );
      // c = [100, 100, 100, 100]
    }

Reduction
---------

:Source: `src/cuda/thrust-reduce <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/thrust-reduce>`_

``thrust::reduce`` combines all elements into a single value using a binary operation.
Common reductions include sum, product, min, and max. For more complex reductions,
``thrust::transform_reduce`` combines a transformation with reduction in a single pass,
avoiding intermediate storage.

.. code-block:: cpp

    #include <thrust/device_vector.h>
    #include <thrust/functional.h>
    #include <thrust/reduce.h>

    int main() {
      thrust::device_vector<int> d_vec(4);
      d_vec[0] = 1;
      d_vec[1] = 2;
      d_vec[2] = 3;
      d_vec[3] = 4;

      // Sum all elements (default)
      int sum = thrust::reduce(d_vec.begin(), d_vec.end());
      // sum = 10

      // Sum with initial value
      int sum_plus_100 = thrust::reduce(d_vec.begin(), d_vec.end(), 100);
      // sum_plus_100 = 110

      // Product of all elements
      int product = thrust::reduce(d_vec.begin(), d_vec.end(), 1, thrust::multiplies<int>());
      // product = 24

      // Find maximum
      int max_val = thrust::reduce(d_vec.begin(), d_vec.end(), INT_MIN, thrust::maximum<int>());
      // max_val = 4
    }

Sorting
-------

:Source: `src/cuda/thrust-sort <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/thrust-sort>`_

Thrust provides efficient parallel sorting with ``thrust::sort`` for in-place sorting
and ``thrust::sort_by_key`` for sorting key-value pairs. Custom comparators allow
sorting in descending order or by custom criteria.

.. code-block:: cpp

    #include <thrust/device_vector.h>
    #include <thrust/functional.h>
    #include <thrust/sort.h>

    int main() {
      thrust::device_vector<int> d_vec(4);
      d_vec[0] = 30;
      d_vec[1] = 10;
      d_vec[2] = 40;
      d_vec[3] = 20;

      // Sort ascending (default)
      thrust::sort(d_vec.begin(), d_vec.end());
      // d_vec = [10, 20, 30, 40]

      // Sort descending
      thrust::sort(d_vec.begin(), d_vec.end(), thrust::greater<int>());
      // d_vec = [40, 30, 20, 10]

      // Sort by key
      thrust::device_vector<int> keys = {3, 1, 4, 2};
      thrust::device_vector<char> vals = {'c', 'a', 'd', 'b'};
      thrust::sort_by_key(keys.begin(), keys.end(), vals.begin());
      // keys = [1, 2, 3, 4], vals = ['a', 'b', 'c', 'd']
    }

Custom Functors
---------------

:Source: `src/cuda/thrust-custom <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/cuda/thrust-custom>`_

For complex operations, define custom functors as structs with ``__host__ __device__``
operator(). This approach works across all CUDA versions and allows stateful functors
with constructor parameters.

.. code-block:: cpp

    #include <thrust/device_vector.h>
    #include <thrust/transform.h>

    struct saxpy_functor {
      const float a;

      saxpy_functor(float _a) : a(_a) {}

      __host__ __device__ float operator()(float x, float y) const {
        return a * x + y;
      }
    };

    void saxpy(float a, thrust::device_vector<float>& x, thrust::device_vector<float>& y) {
      thrust::transform(x.begin(), x.end(), y.begin(), y.begin(), saxpy_functor(a));
    }

    int main() {
      thrust::device_vector<float> x(4, 1.0f);
      thrust::device_vector<float> y(4, 2.0f);

      saxpy(3.0f, x, y);
      // y = [5.0, 5.0, 5.0, 5.0]  (3*1 + 2 = 5)
    }

See Also
--------

- :doc:`cuda_basics` - CUDA fundamentals
- :doc:`cuda_cpp` - libcu++ for CUDA C++ standard library features
