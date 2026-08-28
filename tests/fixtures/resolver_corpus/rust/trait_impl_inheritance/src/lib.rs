pub trait Draw {
    fn draw(&self) -> u32;
}

pub struct Widget;

impl Draw for Widget {
    fn draw(&self) -> u32 {
        1
    }
}
