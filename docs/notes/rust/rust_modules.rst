=======
Modules
=======

.. meta::
   :description: Rust module system compared to C++. Covers mod, use, pub, crate structure, and visibility rules.
   :keywords: Rust, modules, mod, use, pub, crate, C++ namespaces, headers

.. contents:: Table of Contents
    :backlinks: none

:Source: `src/rust/modules <https://github.com/crazyguitar/cppcheatsheet/tree/master/src/rust/modules>`_

Rust's module system organizes code into hierarchical namespaces. Unlike C++
headers, Rust modules are part of the language with explicit visibility rules.

Module Comparison
-----------------

+---------------------------+---------------------------+
| C++                       | Rust                      |
+===========================+===========================+
| ``#include``              | ``mod`` / ``use``         |
+---------------------------+---------------------------+
| ``namespace``             | ``mod``                   |
+---------------------------+---------------------------+
| Header files (``.h``)     | Module files (``.rs``)    |
+---------------------------+---------------------------+
| ``public:``               | ``pub``                   |
+---------------------------+---------------------------+
| ``private:``              | (default, no keyword)     |
+---------------------------+---------------------------+
| ``using namespace``       | ``use``                   |
+---------------------------+---------------------------+

Defining Modules
----------------

**C++:**

.. code-block:: cpp

    // math.h
    namespace math {
      int add(int a, int b);

      namespace utils {
        int square(int x);
      }
    }

    // math.cpp
    namespace math {
      int add(int a, int b) { return a + b; }

      namespace utils {
        int square(int x) { return x * x; }
      }
    }

**Rust (inline modules):**

.. code-block:: rust

    mod math {
        pub fn add(a: i32, b: i32) -> i32 {
            a + b
        }

        pub mod utils {
            pub fn square(x: i32) -> i32 {
                x * x
            }
        }
    }

    fn main() {
        let sum = math::add(1, 2);
        let sq = math::utils::square(3);
    }

File-Based Modules
------------------

Rust maps modules to files:

.. code-block:: text

    src/
    ├── main.rs
    ├── math.rs          # mod math
    └── math/
        └── utils.rs     # mod math::utils

**src/main.rs:**

.. code-block:: rust

    mod math;  // loads math.rs or math/mod.rs

    fn main() {
        let sum = math::add(1, 2);
        let sq = math::utils::square(3);
    }

**src/math.rs:**

.. code-block:: rust

    pub mod utils;  // loads math/utils.rs

    pub fn add(a: i32, b: i32) -> i32 {
        a + b
    }

**src/math/utils.rs:**

.. code-block:: rust

    pub fn square(x: i32) -> i32 {
        x * x
    }

Visibility (pub)
----------------

.. code-block:: rust

    mod outer {
        pub fn public_fn() {}      // visible outside module
        fn private_fn() {}         // only visible in this module

        pub mod inner {
            pub fn inner_public() {}
            fn inner_private() {}

            // pub(super) - visible to parent module
            pub(super) fn parent_only() {}

            // pub(crate) - visible within crate
            pub(crate) fn crate_only() {}
        }
    }

    fn main() {
        outer::public_fn();           // OK
        // outer::private_fn();       // error: private
        outer::inner::inner_public(); // OK
        // outer::inner::parent_only(); // error: not visible here
    }

Struct Field Visibility
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: rust

    mod shapes {
        pub struct Rectangle {
            pub width: u32,   // public field
            height: u32,      // private field
        }

        impl Rectangle {
            pub fn new(width: u32, height: u32) -> Self {
                Rectangle { width, height }
            }

            pub fn area(&self) -> u32 {
                self.width * self.height
            }
        }
    }

    fn main() {
        let rect = shapes::Rectangle::new(10, 20);
        println!("{}", rect.width);  // OK
        // println!("{}", rect.height);  // error: private
    }

The use Keyword
---------------

**C++:**

.. code-block:: cpp

    using namespace std;
    using std::vector;
    namespace fs = std::filesystem;

**Rust:**

.. code-block:: rust

    // Import single item
    use std::collections::HashMap;

    // Import multiple items
    use std::collections::{HashMap, HashSet};

    // Import all public items (glob)
    use std::collections::*;

    // Rename import
    use std::collections::HashMap as Map;

    // Re-export
    pub use std::collections::HashMap;

use Paths
~~~~~~~~~

.. code-block:: rust

    mod foo {
        pub mod bar {
            pub fn baz() {}
        }
    }

    // Absolute path from crate root
    use crate::foo::bar::baz;

    // Relative path
    use self::foo::bar::baz;

    // Parent module
    use super::something;

    // External crate
    use std::io::Read;

Crate Structure
---------------

.. code-block:: text

    my_crate/
    ├── Cargo.toml
    └── src/
        ├── lib.rs       # crate root for library
        ├── main.rs      # crate root for binary
        ├── module_a.rs
        └── module_b/
            ├── mod.rs   # module_b root
            └── sub.rs   # module_b::sub

**src/lib.rs:**

.. code-block:: rust

    pub mod module_a;
    pub mod module_b;

    // Re-export for convenient access
    pub use module_a::important_fn;

Prelude Pattern
---------------

.. code-block:: rust

    // src/lib.rs
    pub mod prelude {
        pub use crate::module_a::TypeA;
        pub use crate::module_b::TypeB;
        pub use crate::traits::*;
    }

    // Users can import common items easily:
    // use my_crate::prelude::*;

External Crates
---------------

**Cargo.toml:**

.. code-block:: toml

    [dependencies]
    serde = "1.0"
    tokio = { version = "1", features = ["full"] }

**src/main.rs:**

.. code-block:: rust

    use serde::{Serialize, Deserialize};
    use tokio::fs::File;

    #[derive(Serialize, Deserialize)]
    struct Config {
        name: String,
    }

Conditional Compilation
-----------------------

.. code-block:: rust

    // Platform-specific modules
    #[cfg(target_os = "linux")]
    mod linux;

    #[cfg(target_os = "windows")]
    mod windows;

    // Feature-gated modules
    #[cfg(feature = "advanced")]
    pub mod advanced;

    // Test module
    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_something() {
            assert!(true);
        }
    }

See Also
--------

- :doc:`rust_traits` - Trait visibility
- :doc:`rust_error` - Error types across modules
