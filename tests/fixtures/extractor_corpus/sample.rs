use crate::app::store::Saver;
use crate::app::{util::limit, fmt::show};
use crate::app::util::*;
use super::helper::assist;
use self::inner::Local;
use std::io::Result as IoResult;

pub use crate::app::api::publicate;

mod inner;

struct User {
    name: String,
}

pub struct Widget {
    pub label: String,
    pub bound: Box<dyn Draw>,
}

struct Pair<T> {
    left: T,
    right: Vec<T>,
}

enum State {
    Ready,
    Done(u32),
}

pub trait Runner {
    fn ready(&self) -> bool;
    fn steps(&self) -> u32;
}

impl Draw for Widget {
    fn draw(&self) -> String {
        self.label.clone()
    }
}

impl Runner for Widget {
    fn ready(&self) -> bool {
        self.label.is_empty()
    }
    fn steps(&self) -> u32 {
        let base = self.ready() as u32;
        for part in 0..base {
            let _spare = part;
        }
        base + 1
    }
}

impl Widget {
    fn new(label: String) -> Self {
        Self::blank(&Widget {
            label,
            bound: Box::new(Gadget),
        })
    }
    fn blank(other: &Widget) -> Self {
        let copy = other;
        copy.duplicate()
    }
    fn duplicate(&self) -> Self {
        Widget {
            label: self.label.clone(),
            bound: self.bound,
        }
    }
}

struct Gadget;

trait Draw {
    fn draw(&self) -> String;
}

fn build_user() -> User {
    User {
        name: String::new(),
    }
}

pub fn load(path: &str) -> IoResult<Vec<u8>> {
    let pairs: Pair<u32> = Pair {
        left: limit(),
        right: Vec::new(),
    };
    let widget = Widget::new(path.to_string());
    widget.ready();
    Saver::save(&widget)?;
    let mode = State::Ready;
    assist(mode);
    show(pairs.left);
    publicate(pairs.right);
    Ok(Vec::<u8>::new())
}

fn local_scope() {
    let mut count = 0;
    count += 1;
    let add = |a: u32, b: u32| a + b;
    let _ = add(count, Local::width());
}

const VERSION: u32 = 1;
