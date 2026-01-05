======
String
======

.. contents:: Table of Contents
    :backlinks: none

Character to String Conversion
------------------------------

:Source: `src/string/char-to-string <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/string/char-to-string>`_

Converting a single character to a ``std::string`` is a common operation in
C++. The language provides several approaches, each with different semantics
and use cases. Understanding these options helps write clearer, more intentional
code.

**Using the fill constructor:**

The ``std::string`` constructor ``string(size_t n, char c)`` creates a string
containing ``n`` copies of character ``c``. This is the most explicit approach
when you need to create a string from a single character, as it clearly
communicates the intent:

.. code-block:: cpp

    #include <string>

    int main(int argc, char *argv[]) {
      std::string s(1, 'a');  // Creates "a"
    }

**Using the append operator:**

The ``+=`` operator appends a character to an existing string. This approach
is useful when building strings incrementally or when you already have a
string object that needs modification:

.. code-block:: cpp

    #include <string>

    int main(int argc, char *argv[]) {
      std::string s;
      s += 'a';  // s is now "a"
    }

**Using the assignment operator:**

Direct assignment replaces the string's contents with a single character.
This is concise but less explicit about the intent compared to the constructor
approach:

.. code-block:: cpp

    #include <string>

    int main(int argc, char *argv[]) {
      std::string s;
      s = 'a';  // s is now "a"
    }

C String to std::string
-----------------------

:Source: `src/string/cstr-to-string <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/string/cstr-to-string>`_

Converting C-style strings (null-terminated character arrays) to ``std::string``
is straightforward due to implicit conversion. The ``std::string`` class has
a constructor that accepts a ``const char*``, which handles the conversion
automatically. This makes interoperability between C and C++ code seamless:

.. code-block:: cpp

    #include <string>

    int main(int argc, char *argv[]) {
      char cstr[] = "hello cstr";
      std::string s = cstr;  // Implicit conversion
    }

Substring Extraction
--------------------

:Source: `src/string/substr <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/string/substr>`_

The ``substr(pos, len)`` member function extracts a portion of a string starting
at position ``pos`` with length ``len``. If ``len`` is omitted or exceeds the
remaining characters, it extracts until the end of the string:

.. code-block:: cpp

    #include <iostream>
    #include <string>

    int main(int argc, char *argv[]) {
      std::string s = "Hello World";

      std::cout << s.substr(0, 5) << "\n";  // Output: Hello
      std::cout << s.substr(6) << "\n";     // Output: World
      std::cout << s.substr(6, 3) << "\n";  // Output: Wor
    }

String Splitting
----------------

:Source: `src/string/split <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/string/split>`_

Splitting strings by a delimiter is a fundamental text processing operation.
Unlike some languages, C++ does not provide a built-in split function in the
standard library. However, several approaches exist using standard library
components.

**Using find and substr:**

This approach manually searches for the delimiter and extracts substrings.
It modifies a copy of the input string by erasing processed portions. While
verbose, this method provides full control over the splitting logic:

.. code-block:: cpp

    #include <iostream>
    #include <string>
    #include <vector>

    std::vector<std::string> split(const std::string &str, char delim) {
      std::string s = str;
      std::vector<std::string> out;
      size_t pos = 0;

      while ((pos = s.find(delim)) != std::string::npos) {
        out.emplace_back(s.substr(0, pos));
        s.erase(0, pos + 1);
      }
      out.emplace_back(s);
      return out;
    }

    int main(int argc, char *argv[]) {
      auto v = split("abc,def,ghi", ',');
      for (const auto &c : v) {
        std::cout << c << "\n";
      }
    }

**Using std::getline:**

The ``std::getline`` function can read from a string stream using a custom
delimiter. This approach is concise and leverages the standard library's
stream facilities. It handles edge cases like empty tokens correctly:

.. code-block:: cpp

    #include <iostream>
    #include <sstream>
    #include <string>
    #include <vector>

    int main(int argc, char *argv[]) {
      std::string in = "abc,def,ghi";
      std::vector<std::string> out;
      std::string token;
      std::istringstream stream(in);

      while (std::getline(stream, token, ',')) {
        out.emplace_back(token);
      }

      for (const auto &c : out) {
        std::cout << c << "\n";
      }
    }

String Joining
--------------

:Source: `src/string/join <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/string/join>`_

Joining strings with a delimiter is the inverse of splitting. C++ does not
provide a built-in join function, but it can be implemented using iterators
and ``std::ostringstream`` or simple string concatenation.

**Using ostringstream:**

.. code-block:: cpp

    #include <iostream>
    #include <sstream>
    #include <string>
    #include <vector>

    std::string join(const std::vector<std::string> &v, char delim) {
      std::ostringstream oss;
      for (size_t i = 0; i < v.size(); ++i) {
        if (i > 0) oss << delim;
        oss << v[i];
      }
      return oss.str();
    }

    int main(int argc, char *argv[]) {
      std::vector<std::string> v = {"abc", "def", "ghi"};
      std::cout << join(v, ',') << "\n";  // Output: abc,def,ghi
    }

Case Conversion
---------------

:Source: `src/string/case-conversion <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/string/case-conversion>`_

Converting strings to uppercase or lowercase requires transforming each
character individually. The ``std::transform`` algorithm combined with
``::toupper`` or ``::tolower`` from ``<cctype>`` provides an efficient,
in-place solution. Note that these functions operate on individual characters
and assume ASCII encoding:

.. code-block:: cpp

    #include <algorithm>
    #include <iostream>
    #include <string>

    int main(int argc, char *argv[]) {
      std::string s = "Hello World";

      // Convert to uppercase
      std::transform(s.begin(), s.end(), s.begin(), ::toupper);
      std::cout << s << "\n";  // Output: HELLO WORLD

      // Convert to lowercase
      std::transform(s.begin(), s.end(), s.begin(), ::tolower);
      std::cout << s << "\n";  // Output: hello world
    }

String Concatenation Performance
--------------------------------

:Source: `src/string/concat-perf <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/string/concat-perf>`_

String concatenation performance varies significantly depending on where
characters are added. Appending to the end of a string is an amortized O(1)
operation due to ``std::string``'s internal buffer management. However,
prepending requires shifting all existing characters, making it O(n) per
operation.

This example demonstrates the performance difference. Appending 100,000
characters completes nearly instantly, while prepending the same number
takes significantly longer. Even with ``reserve()``, prepending remains
slow because the data must still be shifted:

.. code-block:: cpp

    #include <chrono>
    #include <iostream>
    #include <string>

    constexpr int total = 100000;

    template <typename F>
    void profile(const char *label, F &&func) {
      const auto start = std::chrono::steady_clock::now();
      func();
      const auto end = std::chrono::steady_clock::now();
      const auto ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      std::cout << label << ": " << ms.count() << " ms\n";
    }

    int main(int argc, char *argv[]) {
      profile("Append", [] {
        std::string s;
        for (int i = 0; i < total; ++i) {
          s += 'a';
        }
      });

      profile("Prepend", [] {
        std::string s;
        for (int i = 0; i < total; ++i) {
          s = std::string(1, 'a') + s;
        }
      });

      profile("Prepend with reserve", [] {
        std::string s;
        s.reserve(total + 1);
        for (int i = 0; i < total; ++i) {
          s = std::string(1, 'a') + s;
        }
      });
    }

String Literals
---------------

:Source: `src/string/literals <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/string/literals>`_

C++14 and C++17 introduced user-defined literals for string types, providing
a more expressive way to create strings. The ``s`` suffix creates a
``std::string``, while ``sv`` creates a ``std::string_view``. These literals
are defined in the ``std::literals`` namespace and help avoid ambiguity
between C strings and C++ string types:

.. code-block:: cpp

    #include <iostream>
    #include <string>
    #include <string_view>

    int main(int argc, char *argv[]) {
      using namespace std::literals;

      auto s1 = "c string";        // const char*
      auto s2 = "std::string"s;    // std::string
      auto s3 = "std::string_view"sv;  // std::string_view

      std::cout << s1 << "\n";
      std::cout << s2 << "\n";
      std::cout << s3 << "\n";
    }

std::string_view
----------------

:Source: `src/string/string-view <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/string/string-view>`_

Introduced in C++17, ``std::string_view`` provides a non-owning reference to
a character sequence. It avoids copying when you only need to read string data,
making it ideal for function parameters that don't need ownership. However,
care must be taken to ensure the underlying data outlives the view.

**Accepting string_view parameters:**

Functions that only read string data should prefer ``std::string_view``
parameters. This accepts both ``std::string`` and C strings without copying:

.. code-block:: cpp

    #include <iostream>
    #include <string_view>

    void print(std::string_view s) {
      std::cout << s << "\n";
    }

    int main(int argc, char *argv[]) {
      const std::string s = "foo";
      print(s);       // Works: string converts to string_view
      print("bar");   // Works: C string converts to string_view
    }

**Converting string_view to string:**

A ``std::string_view`` cannot implicitly convert to ``std::string`` because
this would involve a potentially expensive copy. Explicit conversion is
required to make the cost visible:

.. code-block:: cpp

    #include <iostream>
    #include <string>
    #include <string_view>

    void process(std::string s) {
      std::cout << s << "\n";
    }

    int main(int argc, char *argv[]) {
      std::string_view sv = "foo";
      // process(sv);  // Error: no implicit conversion
      process(std::string(sv));  // OK: explicit conversion
    }

**Null-termination caveat:**

Unlike ``std::string``, a ``std::string_view`` is not guaranteed to be
null-terminated. This is important when interfacing with C APIs that expect
null-terminated strings. The ``data()`` member function returns a pointer
to the underlying character array, but there may be no null terminator:

.. code-block:: cpp

    #include <cstring>
    #include <iostream>
    #include <string_view>

    int main(int argc, char *argv[]) {
      char array[3] = {'B', 'a', 'r'};  // No null terminator
      std::string_view sv(array, sizeof(array));

      // Safe: use size() to limit iteration
      for (size_t i = 0; i < sv.size(); ++i) {
        std::cout << sv[i];
      }
      std::cout << "\n";

      // Dangerous: strlen expects null-terminated string
      // std::cout << std::strlen(sv.data()) << "\n";  // Undefined behavior
    }

.. note::

    When passing ``std::string_view`` to C APIs, first convert to
    ``std::string`` to ensure null-termination, or verify the view
    was created from a null-terminated source.
