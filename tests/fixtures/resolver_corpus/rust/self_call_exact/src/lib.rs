pub struct Widget;

impl Widget {
    pub fn helper(&self) -> u32 {
        1
    }

    pub fn run(&self) -> u32 {
        self.helper()
    }
}
