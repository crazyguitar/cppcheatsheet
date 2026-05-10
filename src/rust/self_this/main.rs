struct Counter {
    value: i32,
}

impl Counter {
    // Associated function (no self) - like static method
    fn new() -> Self {
        Counter { value: 0 }
    }

    // &self - immutable borrow
    fn get(&self) -> i32 {
        self.value
    }

    // &mut self - mutable borrow
    fn increment(&mut self) {
        self.value += 1;
    }

    // self - takes ownership (consumes)
    fn into_value(self) -> i32 {
        self.value
    }
}

fn main() {
    let mut c = Counter::new();
    println!("Initial: {}", c.get());

    c.increment();
    c.increment();
    println!("After increment: {}", c.get());

    let v = c.into_value();
    println!("Consumed value: {}", v);
    // c is no longer valid here
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let c = Counter::new();
        assert_eq!(c.get(), 0);
    }

    #[test]
    fn test_immutable_borrow() {
        let c = Counter { value: 42 };
        assert_eq!(c.get(), 42);
        assert_eq!(c.get(), 42); // can borrow multiple times
    }

    #[test]
    fn test_mutable_borrow() {
        let mut c = Counter::new();
        c.increment();
        c.increment();
        assert_eq!(c.get(), 2);
    }

    #[test]
    fn test_ownership_transfer() {
        let c = Counter { value: 100 };
        let v = c.into_value();
        assert_eq!(v, 100);
        // c is consumed, can't use it anymore
    }
}
