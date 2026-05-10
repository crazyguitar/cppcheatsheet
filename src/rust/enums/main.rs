enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
}

fn process(msg: &Message) -> String {
    match msg {
        Message::Quit => "Quit".to_string(),
        Message::Move { x, y } => format!("Move to ({}, {})", x, y),
        Message::Write(s) => format!("Write: {}", s),
    }
}

fn main() {
    let msgs = vec![
        Message::Quit,
        Message::Move { x: 10, y: 20 },
        Message::Write("Hello".to_string()),
    ];

    for msg in &msgs {
        println!("{}", process(msg));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quit() {
        let msg = Message::Quit;
        assert_eq!(process(&msg), "Quit");
    }

    #[test]
    fn test_move() {
        let msg = Message::Move { x: 10, y: 20 };
        assert_eq!(process(&msg), "Move to (10, 20)");
    }

    #[test]
    fn test_write() {
        let msg = Message::Write("Hello".to_string());
        assert_eq!(process(&msg), "Write: Hello");
    }
}
