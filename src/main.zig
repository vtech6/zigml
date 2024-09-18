const std = @import("std");
const value = @import("./model/value.zig");
pub fn main() !void {
    const a = value.Value.create(2, std.heap.page_allocator);
    const b = value.Value.create(0, std.heap.page_allocator);
    const d = value.Value.create(-3, std.heap.page_allocator);
    const e = value.Value.create(1, std.heap.page_allocator);
    const c = a.multiply(d);
    const f = b.multiply(e);
    const bias = value.Value.create(6.8813735870195432, std.heap.page_allocator);
    const g = c.add(f);
    const h = g.add(bias);
    var i = h.tanh();
    i.backpropagate();
    i.visualize();
}

test "simple test" {
    var list = std.ArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}
