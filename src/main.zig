const std = @import("std");
const value = @import("./model/value.zig");
pub fn main() !void {
    const a = value.Value.create(5, std.heap.page_allocator);
    const b = value.Value.create(3, std.heap.page_allocator);
    const c = a.add(b);
    const d = value.Value.create(42, std.heap.page_allocator);
    const e = value.Value.create(13, std.heap.page_allocator);
    const f = e.add(d);
    const g = c.multiply(f);
    g.visualize();
}

test "simple test" {
    var list = std.ArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}
