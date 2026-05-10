use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let counter = Arc::new(Mutex::new(0));

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let counter = Arc::clone(&counter);
            thread::spawn(move || {
                let mut num = counter.lock().unwrap();
                *num += 1;
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Counter: {}", *counter.lock().unwrap());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_counter() {
        let counter = Arc::new(Mutex::new(0));

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let counter = Arc::clone(&counter);
                thread::spawn(move || {
                    let mut num = counter.lock().unwrap();
                    *num += 1;
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(*counter.lock().unwrap(), 10);
    }

    #[test]
    fn test_thread_spawn() {
        let handle = thread::spawn(|| 42);
        assert_eq!(handle.join().unwrap(), 42);
    }
}
