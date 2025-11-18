======
String
======

.. contents:: Table of Contents
    :backlinks: none

Char to a string
----------------

.. code-block:: cpp

    #include <string>

    int main(int argc, char *argv[]) {
      // string(size_t n, char c)
      std::string a(1, 'a');
    }

.. code-block:: cpp

    #include <string>

    int main(int argc, char *argv[]) {
      std::string s;
      s += 'a';
    }

.. code-block:: cpp

    #include <string>

    int main(int argc, char *argv[]) {
      std::string s;
      s = 'a';
    }

C String to a String
--------------------

.. code-block:: cpp

    #include <string>

    int main(int argc, char *argv[]) {
      char cstr[] = "hello cstr";
      std::string s = cstr;
    }

Split a String
--------------

.. code-block:: cpp

    // $ g++ --std=c++14 -Wall -Werror -g -O3 split.cpp
    // $ ./a.out
    // abc
    // def
    // ghi

    #include <iostream>
    #include <string>
    #include <vector>

    using namespace std;

    vector<string> split(const string &str, char delimeter) {

        string s = str;
        vector<string> out;
        size_t pos = 0;

        while((pos = s.find(delimeter)) != string::npos) {
            string token = s.substr(0, pos);
            out.emplace_back(token);
            s.erase(0, pos + 1);
        }
        out.emplace_back(s);
        return out;
    }

    int main(int argc, char *argv[]) {

        string s = "abc,def,ghi";
        vector<string> v = split(s, ',');
        for (const auto &c : v) {
            cout << c << "\n";
        }
    }

Using istream

.. code-block:: cpp

    #include <iostream>
    #include <sstream>
    #include <string>
    #include <vector>

    using namespace std;

    template<char delimiter>
    class String : public string
    {
        friend istream &operator>>( istream  &is, String &out) {
            std::getline(is, out, delimiter);
            return is;
        }
    };

    int main(int argc, char *argv[]) {
        std::string text = "abc,def,ghi";

        istringstream iss(text);
        vector<string> out((istream_iterator<String<','>>(iss)),
                            istream_iterator<String<','>>());

        for (const auto &c : out) {
            cout << c << "\n";
        }
    }



Using ``std::getline``

.. code-block:: cpp

    #include <iostream>
    #include <sstream>
    #include <string>
    #include <vector>

    using namespace std;

    int main(int argc, char *argv[])
    {
        string in = "abc,def,ghi";
        vector<string> out;
        string token;
        std::istringstream stream(in);

        while (std::getline(stream, token, ',')) {
            out.emplace_back(token);
        }
        for (const auto &c : out) {
            cout << c << "\n";
        }
    }

Using boost

.. code-block:: cpp

    #include <iostream>
    #include <string>
    #include <vector>
    #include <boost/algorithm/string.hpp>

    using namespace std;

    int main(int argc, char *argv[]) {
        string in = "abc,def,ghi";
        vector<string> out;

        boost::split(out, in, [](char c) { return c == ','; });
        for (const auto &s : out) {
            cout << s << "\n";
        }
    }

Upper & Lower
-------------

.. code-block:: cpp

    // cc -std=c++17 -Wall -Werror -O3 a.cpp

    #include <iostream>
    #include <string>
    #include <algorithm>

    int main(int argc, char *argv[])
    {
      std::string s = "Hello World";
      // to upper
      std::transform(s.begin(), s.end(), s.begin(), ::toupper);
      std::cout << s << "\n";

      // to lower
      std::transform(s.begin(), s.end(), s.begin(), ::tolower);
      std::cout << s << "\n";
    }

String Concat
-------------

Note that concatenating a string at the beginning is much slower than appending
in the end. Although reserving space can speed up inserting a string in front of
another one, the performance is still much slower than appending a string at the
back.

.. code-block:: cpp

	#include <iostream>
	#include <chrono>

	constexpr int total = 100000;
	using milliseconds = std::chrono::milliseconds;

	template <typename F>
	void profile(F &&func) {
	  const auto start = std::chrono::steady_clock::now();
	  func();
	  const auto end = std::chrono::steady_clock::now();
	  const auto d = end - start;
	  const auto mill = std::chrono::duration_cast<milliseconds>(d).count();
	  std::cout << mill << " ms\n";
	}

	int main(int argc, char *argv[]) {

	  profile([] {
		std::string s;
		for (int i = 0; i < total; ++i) {
		  s += 'a';
		}
	  });

	  profile([] {
		std::string s;
		for (int i = 0; i < total; ++i) {
		  s = std::string(1, 'a') + s;
		}
	  });

	  profile([] {
	    std::string s;
	    s.reserve(total+1);
	    for (int i = 0; i < total; ++i) {
	      s = std::string(1, 'a') + s;
	    }
	  });
	}

    // $ g++ -std=c++17 -Wall -Werror a.cc
    // 0 ms
    // 143 ms
    // 110 ms

String Literals
---------------

.. code-block:: cpp

    #include <iostream>
    #include <string>
    #include <string_view>

    int main(int argc, char *argv[]) {
      using namespace std::literals;

      auto s1 = "c string";
      auto s2 = "std::string"s;
      auto s3 = "std::string_view"sv;

      std::cout << s1 << "\n";
      std::cout << s2 << "\n";
      std::cout << s3 << "\n";
    }

String View
-----------

.. code-block:: cpp

    #include <iostream>
    #include <string_view>

    void f(std::string_view s) {
      std::cout << s << "\n";
    }

    int main(int argc, char *argv[]) {
      const std::string s = "foo";
      // pass a const string is ok
      f(s);
    }

.. code-block:: cpp

    #include <iostream>
    #include <string_view>

    void f(std::string s) {
      std::cout << s << "\n";
    }

    int main(int argc, char *argv[]) {
      std::string_view s = "foo";
      f(s); // compile error. cannot convert a string_view to a string
    }

.. code-block:: cpp

    #include <iostream>
    #include <string_view>

    void f(std::string s) {
      std::cout << s << "\n";
    }

    int main(int argc, char *argv[]) {
      std::string_view s = "foo";
      // we can cast a string_view to a string
      f(static_cast<std::string>(s));
    }

.. code-block:: cpp

    // string_view is not alway has null-terminated
    #include <iostream>
    #include <cstring>
    #include <string_view>

    int main(int argc, char *argv[]) {
      char array[3] = {'B', 'a', 'r'};
      std::string_view s(array, sizeof array);
      // Dangerous!! ptr will access memory address larger than array+3
      for (auto ptr = s.data(); !!ptr; ++ptr) {
        std::cout << *ptr << "\n";
      }
    }
