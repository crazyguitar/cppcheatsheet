fn main() {
    let x = 5; // immutable
    println!("x = {}", x);

    let mut y = 5; // mutable
    println!("y = {}", y);
    y = 10;
    println!("y = {}", y);

    // shadowing
    let z = 5;
    let z = z + 1;
    println!("z = {}", z);
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_immutable() {
        let x = 5;
        assert_eq!(x, 5);
    }

    #[test]
    fn test_mutable() {
        let mut y = 5;
        y = 10;
        assert_eq!(y, 10);
    }

    #[test]
    fn test_shadowing() {
        let z = 5;
        let z = z + 1;
        assert_eq!(z, 6);
    }
}
