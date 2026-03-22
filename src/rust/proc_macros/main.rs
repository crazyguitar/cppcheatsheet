/// Rust has two kinds of macros:
/// 1. Declarative macros (macro_rules!) - pattern-based code generation
/// 2. Procedural macros - operate on the AST at compile time
///
/// This file demonstrates built-in attributes and declarative macros.
/// Procedural macros must be defined in a separate crate with
/// proc-macro = true in Cargo.toml.

// ---- Built-in Attributes ----

// #[derive] auto-generates trait implementations
#[derive(Debug, Clone, PartialEq)]
struct Point {
    x: f64,
    y: f64,
}

// #[cfg] for conditional compilation (like #ifdef in C++)
#[cfg(target_os = "linux")]
fn platform() -> &'static str {
    "linux"
}

#[cfg(target_os = "macos")]
fn platform() -> &'static str {
    "macos"
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn platform() -> &'static str {
    "other"
}

// #[allow] to suppress warnings (like #pragma in C++)
#[allow(dead_code)]
fn unused_function() -> i32 {
    42
}

// ---- Declarative Macros (macro_rules!) ----

// Simple macro similar to C++ #define but type-safe
macro_rules! say_hello {
    () => {
        println!("Hello!");
    };
    ($name:expr) => {
        println!("Hello, {}!", $name);
    };
}

// Macro with repetition - no C++ equivalent
macro_rules! vec_of_strings {
    ($($x:expr),* $(,)?) => {
        vec![$($x.to_string()),*]
    };
}

// Macro that generates a function
macro_rules! make_adder {
    ($name:ident, $t:ty) => {
        fn $name(a: $t, b: $t) -> $t {
            a + b
        }
    };
}

make_adder!(add_i32, i32);
make_adder!(add_f64, f64);

fn main() {
    // Built-in attributes
    let p = Point { x: 1.0, y: 2.0 };
    let p2 = p.clone();
    println!("{:?}", p); // Debug trait from #[derive]
    println!("same? {}", p == p2); // PartialEq trait from #[derive]
    println!("platform: {}", platform());

    // Declarative macros
    say_hello!();
    say_hello!("Rust");

    let v = vec_of_strings!["a", "b", "c"];
    println!("{:?}", v);

    println!("add_i32: {}", add_i32(1, 2));
    println!("add_f64: {}", add_f64(1.0, 2.0));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_debug() {
        let p = Point { x: 1.0, y: 2.0 };
        assert_eq!(format!("{:?}", p), "Point { x: 1.0, y: 2.0 }");
    }

    #[test]
    fn test_derive_clone_partialeq() {
        let p = Point { x: 1.0, y: 2.0 };
        let p2 = p.clone();
        assert_eq!(p, p2);
    }

    #[test]
    fn test_cfg() {
        let os = platform();
        assert!(!os.is_empty());
    }

    #[test]
    fn test_vec_of_strings() {
        let v = vec_of_strings!["a", "b", "c"];
        assert_eq!(v, vec!["a".to_string(), "b".to_string(), "c".to_string()]);
    }

    #[test]
    fn test_make_adder() {
        assert_eq!(add_i32(1, 2), 3);
        assert_eq!(add_f64(1.5, 2.5), 4.0);
    }
}
