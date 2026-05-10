use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    // Box: unique ownership on heap
    let boxed = Box::new(42);
    println!("Boxed value: {}", boxed);

    // Rc: reference counted
    let shared = Rc::new(42);
    let shared2 = Rc::clone(&shared);
    println!(
        "Shared value: {}, count: {}",
        shared,
        Rc::strong_count(&shared)
    );
    drop(shared2);
    println!("After drop, count: {}", Rc::strong_count(&shared));

    // RefCell: interior mutability
    let shared_mut = Rc::new(RefCell::new(42));
    *shared_mut.borrow_mut() += 1;
    println!("Mutated value: {}", shared_mut.borrow());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box() {
        let boxed = Box::new(42);
        assert_eq!(*boxed, 42);
    }

    #[test]
    fn test_rc() {
        let shared = Rc::new(42);
        let shared2 = Rc::clone(&shared);
        assert_eq!(Rc::strong_count(&shared), 2);
        drop(shared2);
        assert_eq!(Rc::strong_count(&shared), 1);
    }

    #[test]
    fn test_refcell() {
        let cell = RefCell::new(42);
        *cell.borrow_mut() += 1;
        assert_eq!(*cell.borrow(), 43);
    }
}
