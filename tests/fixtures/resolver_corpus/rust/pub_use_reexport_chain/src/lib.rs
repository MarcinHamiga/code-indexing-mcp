mod api;

use crate::api::Kick;

pub fn run() -> u32 {
    Kick(3)
}
