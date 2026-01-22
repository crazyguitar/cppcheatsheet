// Lifetime annotations ensure references remain valid

// Return owned data to avoid lifetime issues
fn get_string_owned() -> String {
    String::from("hello")
}

// Lifetime annotation: output lives as long as input
fn longer<'a>(a: &'a str, b: &'a str) -> &'a str {
    if a.len() > b.len() {
        a
    } else {
        b
    }
}

// Lifetime elision: compiler infers lifetime
fn first_word(s: &str) -> &str {
    s.split_whitespace().next().unwrap_or("")
}

// Struct holding a reference needs lifetime annotation
struct Excerpt<'a> {
    text: &'a str,
}

impl<'a> Excerpt<'a> {
    fn new(text: &'a str) -> Self {
        Excerpt { text }
    }

    // Method returning reference tied to self's lifetime
    fn get_text(&self) -> &str {
        self.text
    }
}

// Static lifetime: valid for entire program
fn get_greeting() -> &'static str {
    "Hello, world!"
}

// Different lifetimes for different parameters
fn first_only<'a, 'b>(first: &'a str, _second: &'b str) -> &'a str {
    first
}

// Mutable reference with lifetime
fn push_and_get<'a>(vec: &'a mut Vec<i32>, value: i32) -> &'a i32 {
    vec.push(value);
    vec.last().unwrap()
}

fn main() {
    // Owned data
    let s = get_string_owned();
    println!("Owned: {}", s);

    // Lifetime annotation example
    let s1 = String::from("short");
    let s2 = String::from("longer string");
    let result = longer(&s1, &s2);
    println!("Longer: {}", result);

    // Struct with lifetime
    let novel = String::from("Call me Ishmael. Some years ago...");
    let first_sentence = novel.split('.').next().unwrap();
    let excerpt = Excerpt::new(first_sentence);
    println!("Excerpt: {}", excerpt.get_text());

    // Static lifetime
    let greeting = get_greeting();
    println!("{}", greeting);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_longer() {
        let s1 = "short";
        let s2 = "longer string";
        assert_eq!(longer(s1, s2), "longer string");
        assert_eq!(longer(s2, s1), "longer string");
    }

    #[test]
    fn test_first_word() {
        assert_eq!(first_word("hello world"), "hello");
        assert_eq!(first_word("single"), "single");
        assert_eq!(first_word(""), "");
    }

    #[test]
    fn test_excerpt_struct() {
        let text = String::from("Hello, Rust!");
        let excerpt = Excerpt::new(&text);
        assert_eq!(excerpt.get_text(), "Hello, Rust!");
    }

    #[test]
    fn test_static_lifetime() {
        let greeting = get_greeting();
        assert_eq!(greeting, "Hello, world!");
    }

    #[test]
    fn test_first_only() {
        let s1 = String::from("first");
        let s2 = String::from("second");
        assert_eq!(first_only(&s1, &s2), "first");
    }

    #[test]
    fn test_push_and_get() {
        let mut vec = vec![1, 2, 3];
        let last = push_and_get(&mut vec, 4);
        assert_eq!(*last, 4);
    }

    #[test]
    fn test_lifetime_in_scope() {
        let result;
        let s1 = String::from("long string");
        {
            let s2 = String::from("short");
            // result must not outlive s2, but s1 lives longer
            result = longer(&s1, &s2);
            assert!(result == "long string" || result == "short");
        }
        // s2 is dropped, but result points to s1 which is still valid
        // Note: This works because result happens to point to s1
    }
}
