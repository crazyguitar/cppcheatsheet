fn main() {
    let x = 10;

    // Closure that borrows x
    let by_ref = || x + 1;
    println!("by_ref: {}", by_ref());

    // Closure that takes ownership
    let s = String::from("hello");
    let by_move = move || s.len();
    println!("by_move: {}", by_move());
    // println!("{}", s); // error: s was moved

    // Mutable closure
    let mut y = 10;
    let mut by_mut = || {
        y += 1;
        y
    };
    println!("by_mut: {}", by_mut());
    println!("by_mut: {}", by_mut());

    // Closure as function parameter
    let nums = vec![1, 2, 3];
    let doubled: Vec<i32> = nums.iter().map(|&n| n * 2).collect();
    println!("doubled: {:?}", doubled);
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_borrow_closure() {
        let x = 10;
        let add_one = || x + 1;
        assert_eq!(add_one(), 11);
    }

    #[test]
    fn test_move_closure() {
        let s = String::from("hello");
        let len = move || s.len();
        assert_eq!(len(), 5);
    }

    #[test]
    fn test_mut_closure() {
        let mut counter = 0;
        let mut inc = || {
            counter += 1;
            counter
        };
        assert_eq!(inc(), 1);
        assert_eq!(inc(), 2);
    }

    #[test]
    fn test_closure_as_param() {
        fn apply<F: Fn(i32) -> i32>(f: F, x: i32) -> i32 {
            f(x)
        }
        let double = |x| x * 2;
        assert_eq!(apply(double, 5), 10);
    }
}
