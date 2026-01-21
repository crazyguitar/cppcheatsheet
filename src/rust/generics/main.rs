fn max<T: PartialOrd>(a: T, b: T) -> T {
    if a > b {
        a
    } else {
        b
    }
}

fn max_where<T>(a: T, b: T) -> T
where
    T: PartialOrd,
{
    if a > b {
        a
    } else {
        b
    }
}

// Generic struct
struct Pair<T> {
    first: T,
    second: T,
}

impl<T> Pair<T> {
    fn new(first: T, second: T) -> Self {
        Pair { first, second }
    }
}

impl<T: PartialOrd> Pair<T> {
    fn max(&self) -> &T {
        if self.first > self.second {
            &self.first
        } else {
            &self.second
        }
    }
}

fn main() {
    println!("max(3, 5) = {}", max(3, 5));
    println!("max(3.14, 2.71) = {}", max_where(3.14, 2.71));

    let pair = Pair::new(10, 20);
    println!("max of pair = {}", pair.max());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_int() {
        assert_eq!(max(3, 5), 5);
        assert_eq!(max(10, 2), 10);
    }

    #[test]
    fn test_max_float() {
        let result: f64 = max(3.14, 2.71);
        assert!((result - 3.14).abs() < 1e-10);
    }

    #[test]
    fn test_pair_max() {
        let pair = Pair::new(10, 20);
        assert_eq!(*pair.max(), 20);
    }
}
