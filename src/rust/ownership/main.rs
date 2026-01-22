// Ownership: move vs copy, consuming vs borrowing patterns

fn main() {
    // Move semantics
    let s1 = String::from("hello");
    let s2 = s1; // move
    println!("s2 = {}", s2);

    // Explicit clone
    let s3 = s2.clone();
    println!("s2 = {}, s3 = {}", s2, s3);

    // Copy types (stack-only data)
    let x = 5;
    let y = x; // copy, not move
    println!("x = {}, y = {}", x, y);

    // Ownership patterns
    let s = String::from("owned");
    consume(s);
    // println!("{}", s); // Error: s was moved

    let s = String::from("borrowed");
    borrow(&s);
    println!("still valid: {}", s);

    let mut s = String::from("mutable");
    modify(&mut s);
    println!("modified: {}", s);

    let s = create();
    println!("created: {}", s);
}

// Takes ownership - caller loses access
fn consume(s: String) {
    println!("consumed: {}", s);
} // s dropped here

// Borrows - caller retains ownership
fn borrow(s: &String) {
    println!("borrowed: {}", s);
}

// Mutable borrow - temporary exclusive access
fn modify(s: &mut String) {
    s.push_str(" borrow");
}

// Returns ownership to caller
fn create() -> String {
    String::from("new string")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_move() {
        let s1 = String::from("hello");
        let s2 = s1;
        assert_eq!(s2, "hello");
    }

    #[test]
    fn test_clone() {
        let s1 = String::from("hello");
        let s2 = s1.clone();
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_copy_types() {
        let x = 5;
        let y = x;
        assert_eq!(x, 5);
        assert_eq!(y, 5);
    }

    #[test]
    fn test_borrow() {
        let s = String::from("test");
        borrow(&s);
        assert_eq!(s, "test"); // still valid
    }

    #[test]
    fn test_mutable_borrow() {
        let mut s = String::from("hello");
        modify(&mut s);
        assert_eq!(s, "hello borrow");
    }

    #[test]
    fn test_create() {
        let s = create();
        assert_eq!(s, "new string");
    }
}
