==============
Initialization
==============

.. contents:: Table of Contents
    :backlinks: none

Initializer lists
-----------------

.. code-block:: cpp

    #include <iostream>
    #include <initializer_list>

    template<typename T>
    decltype(auto) sum(const std::initializer_list<T> &v) {
        T s = 0;
        for (const auto &i : v) {
            s += i;
        }
        return s;
    }

    int main(int argc, char *argv[]) {
        sum<int>({1,2,3,4,5});
    }
