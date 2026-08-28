pub mod inner;
pub mod outer;
pub mod wrap;

use self::inner::ping;

pub fn run() -> u32 {
    ping(1)
}
