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
}
