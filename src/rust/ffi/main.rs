//! FFI (Foreign Function Interface) - Calling C++ from Rust
//!
//! Demonstrates binding to C++ libraries using extern "C" functions,
//! raw pointers, and safe Rust wrappers.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

// FFI declarations matching cpp_lib.cpp
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

extern "C" {
    fn cpp_add(a: i32, b: i32) -> i32;
    fn cpp_fill_array(arr: *mut i32, len: usize, value: i32);
    fn cpp_create_greeting(name: *const c_char) -> *mut c_char;
    fn cpp_free_string(s: *mut c_char);
    fn cpp_distance(p1: *const Point, p2: *const Point) -> f64;
    fn cpp_midpoint(p1: *const Point, p2: *const Point) -> Point;
}

// Safe Rust wrappers

pub fn add(a: i32, b: i32) -> i32 {
    unsafe { cpp_add(a, b) }
}

pub fn fill_array(arr: &mut [i32], value: i32) {
    unsafe { cpp_fill_array(arr.as_mut_ptr(), arr.len(), value) }
}

pub fn create_greeting(name: &str) -> String {
    let c_name = CString::new(name).expect("CString::new failed");
    unsafe {
        let ptr = cpp_create_greeting(c_name.as_ptr());
        let result = CStr::from_ptr(ptr).to_string_lossy().into_owned();
        cpp_free_string(ptr);
        result
    }
}

pub fn distance(p1: &Point, p2: &Point) -> f64 {
    unsafe { cpp_distance(p1, p2) }
}

pub fn midpoint(p1: &Point, p2: &Point) -> Point {
    unsafe { cpp_midpoint(p1, p2) }
}

fn main() {
    // Simple function call
    let sum = add(10, 20);
    println!("cpp_add(10, 20) = {}", sum);

    // Array manipulation
    let mut arr = [0i32; 5];
    fill_array(&mut arr, 42);
    println!("After cpp_fill_array: {:?}", arr);

    // String handling
    let greeting = create_greeting("Rust");
    println!("cpp_create_greeting: {}", greeting);

    // Struct passing
    let p1 = Point { x: 0.0, y: 0.0 };
    let p2 = Point { x: 3.0, y: 4.0 };
    println!("Distance: {}", distance(&p1, &p2));
    println!("Midpoint: {:?}", midpoint(&p1, &p2));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
        assert_eq!(add(-1, 1), 0);
    }

    #[test]
    fn test_fill_array() {
        let mut arr = [0i32; 3];
        fill_array(&mut arr, 7);
        assert_eq!(arr, [7, 7, 7]);
    }

    #[test]
    fn test_greeting() {
        assert_eq!(create_greeting("World"), "Hello, World!");
    }

    #[test]
    fn test_distance() {
        let p1 = Point { x: 0.0, y: 0.0 };
        let p2 = Point { x: 3.0, y: 4.0 };
        assert!((distance(&p1, &p2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_midpoint() {
        let p1 = Point { x: 0.0, y: 0.0 };
        let p2 = Point { x: 4.0, y: 6.0 };
        let mid = midpoint(&p1, &p2);
        assert!((mid.x - 2.0).abs() < 1e-10);
        assert!((mid.y - 3.0).abs() < 1e-10);
    }
}
