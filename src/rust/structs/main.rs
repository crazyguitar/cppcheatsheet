struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }

    fn distance(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

fn main() {
    let p = Point::new(3.0, 4.0);
    println!("Distance from origin: {}", p.distance());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let p = Point::new(1.0, 2.0);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, 2.0);
    }

    #[test]
    fn test_distance() {
        let p = Point::new(3.0, 4.0);
        assert!((p.distance() - 5.0).abs() < 1e-10);
    }
}
