use std::collections::HashMap;

fn do_something() -> i32 {
    42
}

fn main() {
    // Partial type inference with _
    let v: Vec<_> = vec![1, 2, 3];
    println!("v = {:?}", v);

    let m: HashMap<_, _> = vec![
        ("key".to_string(), 1),
    ].into_iter().collect();
    println!("m = {:?}", m);

    let squares: Vec<_> = (0..5).map(|x| x * x).collect();
    println!("squares = {:?}", squares);

    // Ignoring values in patterns
    let (x, _) = (1, 2);
    println!("x = {}", x);

    let value = Some(42);
    match value {
        Some(_) => println!("has a value"),
        None => println!("empty"),
    }

    // Prefix with _ to suppress unused warning
    let _result = do_something();
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    #[test]
    fn test_partial_type_inference() {
        let v: Vec<_> = vec![1, 2, 3];
        assert_eq!(v, vec![1, 2, 3]);

        let m: HashMap<_, _> = vec![
            ("a".to_string(), 1),
        ].into_iter().collect();
        assert_eq!(m["a"], 1);

        let squares: Vec<_> = (0..5).map(|x| x * x).collect();
        assert_eq!(squares, vec![0, 1, 4, 9, 16]);
    }

    #[test]
    fn test_ignore_in_patterns() {
        let (x, _) = (1, 2);
        assert_eq!(x, 1);

        let value = Some(42);
        let has_value = match value {
            Some(_) => true,
            None => false,
        };
        assert!(has_value);
    }

    #[test]
    fn test_ignore_struct_fields() {
        struct Point { x: i32, _y: i32, _z: i32 }
        let p = Point { x: 1, _y: 2, _z: 3 };
        let Point { x, .. } = p;
        assert_eq!(x, 1);
    }
}
