// Borrowing rules: multiple readers OR single writer

fn modify(x: &mut i32) {
    *x += 1;
}

fn read(x: &i32) {
    println!("{}", x);
}

fn main() {
    let mut val = 5;
    modify(&mut val);
    read(&val);
    println!("val = {}", val);

    // Multiple immutable borrows OK
    let s = String::from("hello");
    let r1 = &s;
    let r2 = &s;
    println!("{} {}", r1, r2);

    // Mutable borrow after immutable borrows end
    let mut s = String::from("hello");
    {
        let r1 = &s;
        let r2 = &s;
        println!("{} {}", r1, r2);
    } // r1, r2 go out of scope
    let r3 = &mut s;
    r3.push_str(" world");
    println!("{}", r3);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutable_borrow() {
        let mut val = 5;
        modify(&mut val);
        assert_eq!(val, 6);
    }

    #[test]
    fn test_immutable_borrow() {
        let val = 10;
        let r = &val;
        assert_eq!(*r, 10);
    }

    #[test]
    fn test_multiple_immutable_borrows() {
        let val = 10;
        let r1 = &val;
        let r2 = &val;
        assert_eq!(*r1 + *r2, 20);
    }

    #[test]
    fn test_borrow_then_mutate() {
        let mut s = String::from("hello");
        {
            let r = &s;
            assert_eq!(r, "hello");
        }
        s.push_str(" world");
        assert_eq!(s, "hello world");
    }

    #[test]
    fn test_mutable_borrow_exclusive() {
        let mut v = vec![1, 2, 3];
        let r = &mut v;
        r.push(4);
        assert_eq!(r.len(), 4);
    }
}
