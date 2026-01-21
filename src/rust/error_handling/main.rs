fn find_value(key: i32) -> Option<i32> {
    if key > 0 {
        Some(key * 2)
    } else {
        None
    }
}

fn divide(a: i32, b: i32) -> Result<i32, String> {
    if b == 0 {
        Err("division by zero".to_string())
    } else {
        Ok(a / b)
    }
}

fn main() {
    // Option
    if let Some(v) = find_value(5) {
        println!("Found: {}", v);
    }

    // Result
    match divide(10, 2) {
        Ok(result) => println!("10 / 2 = {}", result),
        Err(e) => println!("Error: {}", e),
    }

    // Using unwrap_or
    let val = find_value(-1).unwrap_or(0);
    println!("Value or default: {}", val);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_value_some() {
        assert_eq!(find_value(5), Some(10));
    }

    #[test]
    fn test_find_value_none() {
        assert_eq!(find_value(-1), None);
    }

    #[test]
    fn test_divide_ok() {
        assert_eq!(divide(10, 2), Ok(5));
    }

    #[test]
    fn test_divide_err() {
        assert!(divide(10, 0).is_err());
    }
}
