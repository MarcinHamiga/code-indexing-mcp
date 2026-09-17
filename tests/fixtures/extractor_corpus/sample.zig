const std = @import("std");
const store = @import("store.zig");

const Point = struct {
    x: i32,
    y: i32,
};

const VERSION: u32 = 1;

pub fn add(a: i32, b: i32) i32 {
    return a + b;
}

test "basic" {
    try std.testing.expect(add(1, 2) == 3);
}
