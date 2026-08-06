struct User {
    name: String,
}

enum State {
    Ready,
    Done,
}

fn build_user() -> User {
    User {
        name: String::new(),
    }
}

const VERSION: u32 = 1;
