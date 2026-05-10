fn main() {
    // if as expression
    let x = 42;
    let msg = if x == 42 { "found it" } else { "nope" };
    println!("{}", msg);

    // loop with break value
    let mut counter = 0;
    let result = loop {
        counter += 1;
        if counter == 10 {
            break counter * 2;
        }
    };
    println!("loop result: {}", result);

    // for range
    for i in 0..5 {
        print!("{} ", i);
    }
    println!();

    // for inclusive range
    for i in 0..=5 {
        print!("{} ", i);
    }
    println!();

    // expression block
    let val = {
        let a = 10;
        let b = 32;
        a + b // no semicolon = return value
    };
    println!("block result: {}", val);

    // match
    let n = 42;
    match n {
        0..=41 => println!("too small"),
        42 => println!("perfect"),
        _ => println!("too big"),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_if_expression() {
        let x = 42;
        let r = if x > 0 { "positive" } else { "non-positive" };
        assert_eq!(r, "positive");
    }

    #[test]
    fn test_loop_break_value() {
        let mut i = 0;
        let r = loop {
            i += 1;
            if i == 5 {
                break i;
            }
        };
        assert_eq!(r, 5);
    }

    #[test]
    fn test_expression_block() {
        let v = { 1 + 2 };
        assert_eq!(v, 3);
    }
}
