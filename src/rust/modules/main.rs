mod math;

// Inline nested module
mod outer {
    pub mod inner {
        pub fn greet() -> &'static str {
            "Hello!"
        }
    }
}

fn main() {
    let sum = math::add(2, 3);
    let product = math::multiply(2, 3);
    println!("2 + 3 = {}", sum);
    println!("2 * 3 = {}", product);
    println!("{}", outer::inner::greet());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(math::add(2, 3), 5);
    }

    #[test]
    fn test_multiply() {
        assert_eq!(math::multiply(2, 3), 6);
    }

    #[test]
    fn test_nested_module() {
        assert_eq!(outer::inner::greet(), "Hello!");
    }
}
