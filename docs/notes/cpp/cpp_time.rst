====
Time
====

.. contents:: Table of Contents
    :backlinks: none

The ``<chrono>`` library, introduced in C++11, provides a type-safe way to
work with time durations, clocks, and time points. It eliminates common errors
from mixing time units and offers high-resolution timing for performance
measurement. C++20 added calendar and time zone support, making ``<chrono>``
a complete solution for date and time handling.

Getting Current Timestamp
-------------------------

:Source: `src/time/timestamp <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/time/timestamp>`_

The ``system_clock`` provides wall-clock time that can be converted to calendar
time. The ``time_since_epoch()`` method returns the duration since the Unix
epoch (January 1, 1970). Use ``duration_cast`` to convert between time units.

.. code-block:: cpp

    #include <chrono>
    #include <iostream>

    int main() {
      auto now = std::chrono::system_clock::now();
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
          now.time_since_epoch());
      std::cout << ms.count() << "\n";  // Milliseconds since epoch
    }

Duration and Time Arithmetic
----------------------------

:Source: `src/time/duration <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/time/duration>`_

Durations represent time spans with compile-time unit safety. The library
provides predefined durations (``hours``, ``minutes``, ``seconds``, ``milliseconds``,
``microseconds``, ``nanoseconds``) and supports arithmetic operations. C++14
added user-defined literals for convenient duration creation.

.. code-block:: cpp

    #include <chrono>
    #include <iostream>

    int main() {
      using namespace std::chrono_literals;

      auto duration = 2h + 30min + 45s;  // 2 hours, 30 minutes, 45 seconds
      auto total_seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
      std::cout << total_seconds.count() << " seconds\n";  // Output: 9045 seconds

      // Time arithmetic
      auto now = std::chrono::system_clock::now();
      auto future = now + 24h;  // 24 hours from now
    }

Measuring Elapsed Time (Profiling)
----------------------------------

:Source: `src/time/profiling <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/time/profiling>`_

Use ``steady_clock`` for measuring elapsed time because it is monotonic and
not affected by system clock adjustments. ``system_clock`` can jump forward
or backward due to NTP synchronization or manual changes, making it unsuitable
for benchmarking.

.. code-block:: cpp

    #include <chrono>
    #include <iostream>
    #include <thread>

    int main() {
      auto start = std::chrono::steady_clock::now();

      std::this_thread::sleep_for(std::chrono::milliseconds(100));

      auto end = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      std::cout << "Elapsed: " << elapsed.count() << " ms\n";
    }

Converting to time_t
--------------------

:Source: `src/time/time-t <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/time/time-t>`_

The ``to_time_t`` function converts a ``time_point`` to the C-style ``time_t``
type, enabling interoperability with legacy C time functions like ``gmtime``,
``localtime``, and ``strftime``.

.. code-block:: cpp

    #include <chrono>
    #include <ctime>
    #include <iomanip>
    #include <iostream>

    int main() {
      auto now = std::chrono::system_clock::now();
      std::time_t t = std::chrono::system_clock::to_time_t(now);
      std::cout << std::put_time(std::gmtime(&t), "%Y-%m-%d %H:%M:%S UTC") << "\n";
    }

Converting from Timestamp to time_point
---------------------------------------

:Source: `src/time/from-timestamp <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/time/from-timestamp>`_

To convert a Unix timestamp (milliseconds or seconds since epoch) back to a
``time_point``, construct a duration and use it to initialize the time point.

.. code-block:: cpp

    #include <chrono>
    #include <ctime>
    #include <iomanip>
    #include <iostream>

    int main() {
      using namespace std::chrono_literals;

      // Convert milliseconds timestamp to time_point
      auto timestamp_ms = 1602207217323ms;
      std::chrono::system_clock::time_point tp(timestamp_ms);

      std::time_t t = std::chrono::system_clock::to_time_t(tp);
      std::cout << std::put_time(std::gmtime(&t), "%FT%TZ") << "\n";
    }

ISO 8601 Date Format
--------------------

:Source: `src/time/iso8601 <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/time/iso8601>`_

ISO 8601 is the international standard for date and time representation.
Use ``std::put_time`` with format specifiers to produce ISO 8601 compliant
strings. The ``%F`` specifier is equivalent to ``%Y-%m-%d`` and ``%T`` is
equivalent to ``%H:%M:%S``.

.. code-block:: cpp

    #include <chrono>
    #include <ctime>
    #include <iomanip>
    #include <iostream>

    int main() {
      auto now = std::chrono::system_clock::now();
      std::time_t t = std::chrono::system_clock::to_time_t(now);

      // ISO 8601 formats
      std::cout << std::put_time(std::gmtime(&t), "%Y-%m-%dT%H:%M:%SZ") << "\n";
      std::cout << std::put_time(std::gmtime(&t), "%FT%TZ") << "\n";  // Shorthand
    }

Formatting with Time Zones
--------------------------

:Source: `src/time/timezone <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/time/timezone>`_

Before C++20, time zone handling required platform-specific code or the
``TZ`` environment variable. Use ``gmtime`` for UTC and ``localtime`` for
the system's local time zone.

.. code-block:: cpp

    #include <cstdlib>
    #include <ctime>
    #include <iomanip>
    #include <iostream>

    int main() {
      std::time_t t = std::time(nullptr);

      std::cout << "UTC:   " << std::put_time(std::gmtime(&t), "%F %T") << "\n";
      std::cout << "Local: " << std::put_time(std::localtime(&t), "%F %T %Z") << "\n";
    }

C++20 Calendar and Time Zones
-----------------------------

C++20 introduced comprehensive calendar support with ``year``, ``month``,
``day``, ``year_month_day``, and time zone handling via ``std::chrono::zoned_time``.
This eliminates the need for C-style time functions and environment variables.

.. code-block:: cpp

    #include <chrono>
    #include <iostream>

    int main() {
      using namespace std::chrono;

      // Calendar types (C++20)
      auto today = year{2024}/month{3}/day{15};
      std::cout << today << "\n";  // 2024-03-15

      // Time zones (C++20)
      auto now = system_clock::now();
      zoned_time zt{"America/New_York", now};
      std::cout << zt << "\n";
    }

Clock Types Comparison
----------------------

C++ provides three clock types for different use cases:

- ``system_clock``: Wall-clock time, can be converted to calendar time. May
  jump due to NTP or manual adjustments.

- ``steady_clock``: Monotonic clock, never decreases. Best for measuring
  elapsed time and timeouts.

- ``high_resolution_clock``: Highest available resolution. May be an alias
  for ``system_clock`` or ``steady_clock`` depending on implementation.

.. code-block:: cpp

    #include <chrono>
    #include <iostream>

    int main() {
      // Check clock properties
      std::cout << "system_clock is steady: "
                << std::chrono::system_clock::is_steady << "\n";  // Usually false
      std::cout << "steady_clock is steady: "
                << std::chrono::steady_clock::is_steady << "\n";  // Always true
    }

Duration Rounding: floor, ceil, round (C++17)
---------------------------------------------

:Source: `src/time/rounding <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/time/rounding>`_

C++17 added ``floor``, ``ceil``, and ``round`` functions for rounding durations
to specific units. Unlike ``duration_cast`` which truncates toward zero, these
functions provide explicit control over rounding behavior.

.. code-block:: cpp

    #include <chrono>
    #include <iostream>

    int main() {
      using namespace std::chrono;
      using namespace std::chrono_literals;

      auto d = 2700ms;  // 2.7 seconds

      auto floored = floor<seconds>(d);  // 2s (round down)
      auto ceiled = ceil<seconds>(d);    // 3s (round up)
      auto rounded = round<seconds>(d);  // 3s (round to nearest)

      std::cout << "floor: " << floored.count() << "s\n";
      std::cout << "ceil:  " << ceiled.count() << "s\n";
      std::cout << "round: " << rounded.count() << "s\n";
    }

hh_mm_ss: Breaking Down Duration (C++20)
----------------------------------------

:Source: `src/time/hh-mm-ss <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/time/hh-mm-ss>`_

The ``hh_mm_ss`` class template splits a duration into hours, minutes, seconds,
and subseconds. This is useful for displaying durations in human-readable format
without manual arithmetic.

.. code-block:: cpp

    #include <chrono>
    #include <iostream>

    int main() {
      using namespace std::chrono;
      using namespace std::chrono_literals;

      auto d = 3h + 25min + 45s + 123ms;
      hh_mm_ss hms{d};

      std::cout << hms.hours().count() << "h "
                << hms.minutes().count() << "m "
                << hms.seconds().count() << "s "
                << hms.subseconds().count() << "ms\n";
      // Output: 3h 25m 45s 123ms
    }

Timer Utility Class
-------------------

:Source: `src/time/timer <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/time/timer>`_

A reusable RAII-based timer class simplifies profiling by automatically
measuring elapsed time. This pattern is commonly used for benchmarking
code sections.

.. code-block:: cpp

    #include <chrono>
    #include <iostream>
    #include <string>

    class Timer {
     public:
      Timer(std::string name) : name_(std::move(name)), start_(std::chrono::steady_clock::now()) {}

      ~Timer() {
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);
        std::cout << name_ << ": " << ms.count() << " ms\n";
      }

     private:
      std::string name_;
      std::chrono::steady_clock::time_point start_;
    };

    int main() {
      {
        Timer t("loop");
        volatile int sum = 0;
        for (int i = 0; i < 1000000; ++i) sum += i;
      }  // Timer prints elapsed time when destroyed
    }

C++20 Formatting with std::format
---------------------------------

C++20 introduced ``std::format`` support for chrono types, providing a modern
alternative to ``std::put_time``. The format specifiers follow the same pattern
as ``strftime`` but with cleaner syntax.

.. code-block:: cpp

    #include <chrono>
    #include <format>
    #include <iostream>

    int main() {
      using namespace std::chrono;

      auto now = system_clock::now();

      // Format time_point directly (C++20)
      std::cout << std::format("{:%Y-%m-%d %H:%M:%S}", now) << "\n";
      std::cout << std::format("{:%F %T}", now) << "\n";

      // Format duration
      auto d = 3h + 25min + 10s;
      std::cout << std::format("{:%H:%M:%S}", d) << "\n";  // 03:25:10
    }

C++20 Additional Clocks
-----------------------

C++20 added several new clock types for specialized use cases:

- ``utc_clock``: UTC time including leap seconds
- ``tai_clock``: International Atomic Time
- ``gps_clock``: GPS time
- ``file_clock``: File system time (for ``std::filesystem``)

.. code-block:: cpp

    #include <chrono>
    #include <iostream>

    int main() {
      using namespace std::chrono;

      auto sys_now = system_clock::now();
      auto utc_now = utc_clock::now();

      // Convert between clocks
      auto utc_from_sys = utc_clock::from_sys(sys_now);

      std::cout << "System: " << sys_now << "\n";
      std::cout << "UTC:    " << utc_now << "\n";
    }
