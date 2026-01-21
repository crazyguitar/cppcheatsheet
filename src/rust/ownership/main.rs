fn main() {
    let s1 = String::from("hello");
    let s2 = s1.clone(); // explicit clone
    println!("s2 = {}", s2);

    let s3 = String::from("world");
    let s4 = s3; // move
    println!("s4 = {}", s4);
    // println!("{}", s3); // error: value borrowed after move

    // Copy types (stack-only data)
    let x = 5;
    let y = x; // copy, not move
    println!("x = {}, y = {}", x, y);
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_clone() {
        let s1 = String::from("hello");
        let s2 = s1.clone();
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_move() {
        let s1 = String::from("hello");
        let s2 = s1;
        assert_eq!(s2, "hello");
    }

    #[test]
    fn test_copy_types() {
        let x = 5;
        let y = x;
        assert_eq!(x, 5);
        assert_eq!(y, 5);
    }
}
