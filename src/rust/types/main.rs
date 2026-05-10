fn main() {
    // Integer types
    let a: i32 = -42;
    let b: u64 = 1_000_000;
    let c = 0xff_u8;
    let d = 0b1010_u8;

    println!("i32: {}, u64: {}, hex: {}, bin: {}", a, b, c, d);

    // Floating point
    let pi: f64 = 3.14159;
    let e: f32 = 2.718;
    println!("f64: {}, f32: {}", pi, e);

    // Bool and char
    let flag: bool = true;
    let ch: char = '🦀';
    println!("bool: {}, char: {}", flag, ch);

    // Type suffix
    let x = 42u32;
    let y = 3.14f64;
    println!("u32: {}, f64: {}", x, y);
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_integer_types() {
        let a: i8 = -1;
        let b: u8 = 255;
        assert_eq!(a as i16 + b as i16, 254);
    }

    #[test]
    fn test_type_suffix() {
        let x = 42u32;
        assert_eq!(std::mem::size_of_val(&x), 4);
    }
}
