fn print(s: &str) {
    println!("{}", s);
}

fn main() {
    // String (owned) vs &str (borrowed)
    let owned = String::from("hello");
    let literal: &str = "world";

    print(&owned);
    print(literal);

    // Concatenation
    let combined = owned + " " + literal;
    println!("{}", combined);

    // String methods
    let s = String::from("  Hello World  ");
    println!("trimmed: '{}'", s.trim());
    println!("upper: {}", s.to_uppercase());
    println!("lower: {}", s.to_lowercase());

    // Splitting
    let csv = "a,b,c";
    for part in csv.split(',') {
        println!("part: {}", part);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_string_from() {
        let s = String::from("hello");
        assert_eq!(s, "hello");
    }

    #[test]
    fn test_string_concat() {
        let s1 = String::from("hello");
        let s2 = " world";
        let s3 = s1 + s2;
        assert_eq!(s3, "hello world");
    }

    #[test]
    fn test_trim() {
        let s = "  hello  ";
        assert_eq!(s.trim(), "hello");
    }

    #[test]
    fn test_split() {
        let s = "a,b,c";
        let parts: Vec<&str> = s.split(',').collect();
        assert_eq!(parts, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_contains() {
        let s = "hello world";
        assert!(s.contains("world"));
        assert!(!s.contains("foo"));
    }
}
