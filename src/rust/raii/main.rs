struct Resource {
    name: String,
}

impl Resource {
    fn new(name: &str) -> Self {
        println!("Resource '{}' acquired", name);
        Resource { name: name.to_string() }
    }
}

impl Drop for Resource {
    fn drop(&mut self) {
        println!("Resource '{}' released", self.name);
    }
}

fn main() {
    let r1 = Resource::new("first");
    {
        let r2 = Resource::new("second");
        println!("Inside inner scope");
    } // r2 dropped here
    println!("Back in outer scope");
    drop(r1); // explicit drop
    println!("After explicit drop");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drop_order() {
        let _r1 = Resource::new("a");
        let _r2 = Resource::new("b");
        // dropped in reverse order: b, then a
    }

    #[test]
    fn test_explicit_drop() {
        let r = Resource::new("test");
        drop(r);
        // r is no longer valid
    }

    #[test]
    fn test_scope_drop() {
        let outer = Resource::new("outer");
        {
            let _inner = Resource::new("inner");
        }
        assert_eq!(outer.name, "outer");
    }
}
