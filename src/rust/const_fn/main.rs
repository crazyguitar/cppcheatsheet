const fn factorial(n: u64) -> u64 {
    match n {
        0 | 1 => 1,
        _ => n * factorial(n - 1),
    }
}

const FACT_5: u64 = factorial(5);
const FACT_10: u64 = factorial(10);

// const generic
fn create_array<const N: usize>() -> [i32; N] {
    [0; N]
}

// const in struct
struct Buffer<const SIZE: usize> {
    data: [u8; SIZE],
}

impl<const SIZE: usize> Buffer<SIZE> {
    const fn new() -> Self {
        Buffer { data: [0; SIZE] }
    }
}

fn main() {
    println!("5! = {}", FACT_5);
    println!("10! = {}", FACT_10);

    let arr: [i32; 5] = create_array();
    println!("array: {:?}", arr);

    let buf: Buffer<1024> = Buffer::new();
    println!("buffer size: {}", buf.data.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_const_factorial() {
        assert_eq!(FACT_5, 120);
        assert_eq!(FACT_10, 3628800);
    }

    #[test]
    fn test_runtime_factorial() {
        assert_eq!(factorial(6), 720);
    }

    #[test]
    fn test_const_generic_array() {
        let arr: [i32; 3] = create_array();
        assert_eq!(arr, [0, 0, 0]);
    }

    #[test]
    fn test_const_buffer() {
        let buf: Buffer<16> = Buffer::new();
        assert_eq!(buf.data.len(), 16);
    }
}
