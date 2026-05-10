fn main() {
    let v = vec![1, 2, 3, 4, 5];

    // Filter and map (lazy, chained)
    let result: Vec<i32> = v.iter().filter(|&&x| x % 2 == 0).map(|&x| x * 2).collect();
    println!("Filtered and doubled evens: {:?}", result);

    // Sum
    let sum: i32 = v.iter().sum();
    println!("Sum: {}", sum);

    // Find
    let found = v.iter().find(|&&x| x > 3);
    println!("First > 3: {:?}", found);

    // Any/All
    let has_even = v.iter().any(|&x| x % 2 == 0);
    let all_positive = v.iter().all(|&x| x > 0);
    println!("Has even: {}, All positive: {}", has_even, all_positive);
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_filter_map() {
        let v = vec![1, 2, 3, 4, 5];
        let result: Vec<i32> = v.iter().filter(|&&x| x % 2 == 0).map(|&x| x * 2).collect();
        assert_eq!(result, vec![4, 8]);
    }

    #[test]
    fn test_sum() {
        let v = vec![1, 2, 3, 4, 5];
        let sum: i32 = v.iter().sum();
        assert_eq!(sum, 15);
    }

    #[test]
    fn test_find() {
        let v = vec![1, 2, 3, 4, 5];
        assert_eq!(v.iter().find(|&&x| x > 3), Some(&4));
        assert_eq!(v.iter().find(|&&x| x > 10), None);
    }

    #[test]
    fn test_any_all() {
        let v = vec![1, 2, 3, 4, 5];
        assert!(v.iter().any(|&x| x % 2 == 0));
        assert!(v.iter().all(|&x| x > 0));
        assert!(!v.iter().all(|&x| x % 2 == 0));
    }
}
