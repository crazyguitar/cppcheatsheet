// ---- Trait Object (Dynamic Dispatch) ----

trait Shape {
    fn area(&self) -> f64;
    fn name(&self) -> &str;
}

struct Circle {
    radius: f64,
}

impl Shape for Circle {
    fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }
    fn name(&self) -> &str {
        "Circle"
    }
}

struct Rectangle {
    width: f64,
    height: f64,
}

impl Shape for Rectangle {
    fn area(&self) -> f64 {
        self.width * self.height
    }
    fn name(&self) -> &str {
        "Rectangle"
    }
}

// Dynamic dispatch via trait object
fn print_area(shape: &dyn Shape) {
    println!("{}: area = {:.2}", shape.name(), shape.area());
}

// Static dispatch via generics (monomorphized)
fn print_area_static<T: Shape>(shape: &T) {
    println!("{}: area = {:.2}", shape.name(), shape.area());
}

// ---- Enum-based Dispatch ----

enum Animal {
    Dog(String),
    Cat(String),
}

impl Animal {
    fn speak(&self) -> &str {
        match self {
            Animal::Dog(_) => "Woof!",
            Animal::Cat(_) => "Meow!",
        }
    }

    fn name(&self) -> &str {
        match self {
            Animal::Dog(n) | Animal::Cat(n) => n,
        }
    }
}

// ---- Trait Object References (&dyn Trait) ----
//
// &dyn Trait is a "fat pointer": 8 bytes for data ptr + 8 bytes for vtable ptr.
// No heap allocation — it borrows an existing value.
// This is the most lightweight way to do dynamic dispatch.

fn total_area(shapes: &[&dyn Shape]) -> f64 {
    shapes.iter().map(|s| s.area()).sum()
}

// ---- Rc<RefCell<dyn Trait>> — Shared Mutable Trait Objects ----

use std::cell::RefCell;
use std::rc::Rc;

trait Counter {
    fn increment(&mut self);
    fn count(&self) -> u32;
}

struct ClickCounter {
    clicks: u32,
}

impl Counter for ClickCounter {
    fn increment(&mut self) {
        self.clicks += 1;
    }
    fn count(&self) -> u32 {
        self.clicks
    }
}

struct KeyCounter {
    presses: u32,
}

impl Counter for KeyCounter {
    fn increment(&mut self) {
        self.presses += 1;
    }
    fn count(&self) -> u32 {
        self.presses
    }
}

// ---- Rc<RefCell<Box<dyn Trait>>> — shared mutable trait object ----
// When a factory returns Box<dyn Trait>, wrapping it in Rc<RefCell<...>>
// gives shared ownership + interior mutability.

trait Logger {
    fn log(&mut self, msg: &str);
    fn entries(&self) -> &[String];
}

struct ConsoleLogger {
    logs: Vec<String>,
}

impl Logger for ConsoleLogger {
    fn log(&mut self, msg: &str) {
        self.logs.push(msg.to_string());
    }
    fn entries(&self) -> &[String] {
        &self.logs
    }
}

type SharedLogger = Rc<RefCell<Box<dyn Logger>>>;

fn create_logger() -> SharedLogger {
    let logger: Box<dyn Logger> = Box::new(ConsoleLogger { logs: vec![] });
    Rc::new(RefCell::new(logger))
}

// ---- Returning Trait Objects ----

fn make_shape(kind: &str) -> Box<dyn Shape> {
    match kind {
        "circle" => Box::new(Circle { radius: 5.0 }),
        _ => Box::new(Rectangle {
            width: 4.0,
            height: 3.0,
        }),
    }
}

fn main() {
    // Heterogeneous collection via trait objects
    let shapes: Vec<Box<dyn Shape>> = vec![
        Box::new(Circle { radius: 3.0 }),
        Box::new(Rectangle {
            width: 4.0,
            height: 5.0,
        }),
    ];

    for s in &shapes {
        print_area(s.as_ref());
    }

    // Static dispatch
    let c = Circle { radius: 2.0 };
    print_area_static(&c);

    // Enum-based dispatch
    let animals = vec![Animal::Dog("Rex".into()), Animal::Cat("Whiskers".into())];
    for a in &animals {
        println!("{} says {}", a.name(), a.speak());
    }

    // Factory returning trait object
    let s = make_shape("circle");
    print_area(s.as_ref());

    // &dyn Trait references — no heap allocation
    let c2 = Circle { radius: 1.0 };
    let r2 = Rectangle {
        width: 2.0,
        height: 3.0,
    };
    let refs: Vec<&dyn Shape> = vec![&c2, &r2];
    println!("total area = {:.2}", total_area(&refs));

    // Rc<RefCell<dyn Trait>> — shared ownership + interior mutability
    let counters: Vec<Rc<RefCell<dyn Counter>>> = vec![
        Rc::new(RefCell::new(ClickCounter { clicks: 0 })),
        Rc::new(RefCell::new(KeyCounter { presses: 0 })),
    ];

    // Multiple owners can mutate through RefCell
    let shared = Rc::clone(&counters[0]);
    shared.borrow_mut().increment();
    counters[0].borrow_mut().increment();
    println!("click count = {}", counters[0].borrow().count()); // 2

    // Rc<RefCell<Box<dyn Trait>>> — shared ownership + interior mutability
    let logger = create_logger(); // Rc<RefCell<Box<dyn Logger>>>
    let writer = Rc::clone(&logger); // same type, refcount = 2
    let reader = Rc::clone(&logger); // same type, refcount = 3

    // All three are Rc<RefCell<Box<dyn Logger>>> pointing to the same object.
    // RefCell::borrow_mut() enables mutation through shared references.
    writer.borrow_mut().log("hello");
    reader.borrow_mut().log("world");
    println!("log entries = {}", logger.borrow().entries().len()); // 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circle_area() {
        let c = Circle { radius: 1.0 };
        assert!((c.area() - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_rectangle_area() {
        let r = Rectangle {
            width: 3.0,
            height: 4.0,
        };
        assert!((r.area() - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_trait_object_collection() {
        let shapes: Vec<Box<dyn Shape>> = vec![
            Box::new(Circle { radius: 1.0 }),
            Box::new(Rectangle {
                width: 2.0,
                height: 3.0,
            }),
        ];
        assert_eq!(shapes.len(), 2);
    }

    #[test]
    fn test_enum_dispatch() {
        let dog = Animal::Dog("Rex".into());
        assert_eq!(dog.speak(), "Woof!");
        assert_eq!(dog.name(), "Rex");
    }

    #[test]
    fn test_make_shape() {
        let s = make_shape("circle");
        assert_eq!(s.name(), "Circle");
    }

    #[test]
    fn test_ref_trait_objects() {
        let c = Circle { radius: 1.0 };
        let r = Rectangle {
            width: 2.0,
            height: 3.0,
        };
        let refs: Vec<&dyn Shape> = vec![&c, &r];
        let total = total_area(&refs);
        assert!((total - (std::f64::consts::PI + 6.0)).abs() < 1e-10);
    }

    #[test]
    fn test_rc_refcell_dyn() {
        let counter: Rc<RefCell<dyn Counter>> = Rc::new(RefCell::new(ClickCounter { clicks: 0 }));
        let shared = Rc::clone(&counter);

        counter.borrow_mut().increment();
        shared.borrow_mut().increment();

        assert_eq!(counter.borrow().count(), 2);
        assert_eq!(Rc::strong_count(&counter), 2);
    }

    #[test]
    fn test_rc_refcell_box_dyn() {
        let logger = create_logger(); // Rc<RefCell<Box<dyn Logger>>>
        let writer = Rc::clone(&logger); // same type, refcount = 2
        let reader = Rc::clone(&logger); // same type, refcount = 3

        writer.borrow_mut().log("hello"); // RefCell gives &mut access at runtime
        reader.borrow_mut().log("world");

        assert_eq!(logger.borrow().entries().len(), 2);
        assert_eq!(logger.borrow().entries()[0], "hello");
        assert_eq!(Rc::strong_count(&logger), 3);
    }
}
