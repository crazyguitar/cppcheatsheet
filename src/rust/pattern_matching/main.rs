enum Shape {
    Circle(f64),
    Rectangle(f64, f64),
    Triangle { base: f64, height: f64 },
}

fn area(s: &Shape) -> f64 {
    match s {
        Shape::Circle(r) => std::f64::consts::PI * r * r,
        Shape::Rectangle(w, h) => w * h,
        Shape::Triangle { base, height } => 0.5 * base * height,
    }
}

fn describe(val: Option<i32>) -> String {
    match val {
        Some(x) if x > 0 => format!("positive: {}", x),
        Some(0) => "zero".to_string(),
        Some(x) => format!("negative: {}", x),
        None => "nothing".to_string(),
    }
}

fn main() {
    let shapes = vec![
        Shape::Circle(5.0),
        Shape::Rectangle(3.0, 4.0),
        Shape::Triangle {
            base: 6.0,
            height: 3.0,
        },
    ];
    for s in &shapes {
        println!("area = {:.2}", area(s));
    }

    // if let
    let val = Some(42);
    if let Some(x) = val {
        println!("got {}", x);
    }

    // matches! macro
    let n = 42;
    println!("is answer: {}", matches!(n, 42));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circle_area() {
        let s = Shape::Circle(1.0);
        assert!((area(&s) - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_describe() {
        assert_eq!(describe(Some(5)), "positive: 5");
        assert_eq!(describe(Some(0)), "zero");
        assert_eq!(describe(None), "nothing");
    }
}
