const value = @import("value.zig");
const Value = value.Value;
const std = @import("std");
const allocator = std.testing.allocator;
const create = Value.create;

test "value indexing" {
    var a = create(666, allocator);
    const b = create(420, allocator);
    var c = a.add(b);
    try std.testing.expectEqual(true, a.id < b.id);
    try std.testing.expectEqual(true, b.id < c.id);
    c.deinit() catch {};
}
test "value create method" {
    const a = create(666, allocator);
    try std.testing.expectEqual(a.value, 666);
    try std.testing.expectEqual(a.label, "value");
}

test "value add method" {
    var a = create(666, allocator);
    const b = create(420, allocator);
    var c = a.add(b);
    try std.testing.expectEqual(c.value, 666 + 420);
    c.deinit() catch {};
}

test "value struct" {
    var a = create(666, allocator);
    const b = create(420, allocator);
    var c = a.add(b);
    c.rename("c");
    var d = &c;
    try std.testing.expectEqual(d.label, c.label);
    d.rename("d");
    try std.testing.expectEqual(c.label, "d");
    c.deinit() catch {};
}

test "value rename method" {
    var a = create(0, allocator);
    a.rename("a");
    try std.testing.expectEqual(a.label, "a");
}

test "value setChildren method" {
    var a = create(1, allocator);
    var b = create(2, allocator);
    b.rename("b");
    var c = a.add(b);
    std.debug.print("{any}\n", .{c.children.items});
    std.debug.print("{s}\n", .{"This is a message"});
    c.deinit() catch {};
}

test "value buildTopo method" {
    var a = create(1, allocator);
    var b = create(2, allocator);
    b.rename("b");
    var c = a.add(b);

    c.deinit() catch {};
    c.backward() catch {};
}
