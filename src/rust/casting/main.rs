use std::convert::TryInto;

fn main() {
    // Numeric casts with 'as'
    let d: f64 = 3.14;
    let i: i32 = d as i32;
    println!("f64 {} as i32 = {}", d, i);

    // Widening (safe)
    let small: i32 = 42;
    let big: i64 = small.into();
    println!("i32 {} into i64 = {}", small, big);

    // Narrowing (fallible)
    let big: i64 = 1000;
    let result: Result<i8, _> = big.try_into();
    println!("i64 {} try_into i8 = {:?}", big, result);

    // Pointer to integer
    let x = 42;
    let ptr = &x as *const i32;
    let addr = ptr as usize;
    println!("address: {:#x}", addr);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_to_int() {
        let f: f64 = 3.9;
        let i: i32 = f as i32;
        assert_eq!(i, 3); // truncates
    }

    #[test]
    fn test_widening() {
        let small: i8 = 127;
        let big: i32 = small.into();
        assert_eq!(big, 127);
    }

    #[test]
    fn test_narrowing_ok() {
        let big: i32 = 100;
        let small: Result<i8, _> = big.try_into();
        assert_eq!(small, Ok(100));
    }

    #[test]
    fn test_narrowing_overflow() {
        let big: i32 = 1000;
        let small: Result<i8, _> = big.try_into();
        assert!(small.is_err());
    }
}
