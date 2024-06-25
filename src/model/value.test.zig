const value = @import("value.zig");
const std = @import("std");
const create = value.Value.create;

test "value indexing" {
    var a = create(666);
    var b = create(420);
    const c = a.add(&b);
    try std.testing.expectEqual(true, a.id < b.id);
    try std.testing.expectEqual(true, b.id < c.id);
}

test "value create method" {
    const a = create(5);
    try std.testing.expectEqual(a.value, 5);
    try std.testing.expectEqual(a.label, "value");
}

test "value add method" {
    var a = create(5);
    var b = create(3);
    const c = a.add(&b);
    try std.testing.expectEqual(c.value, 8);
}

test "value struct" {
    var a = create(5);
    var b = create(3);
    var c = a.add(&b);
    c.rename("c");
    var d = &c;
    try std.testing.expectEqual(d.label, c.label);
    d.rename("d");
    try std.testing.expectEqual(c.label, "d");
}

test "value rename method" {
    var a = create(0);
    a.rename("a");
    try std.testing.expectEqual(a.label, "a");
}

test "value setChildren method" {
    var a = create(1);
    var b = create(2);
    b.rename("b");
    const c = a.add(&b);
    const children = [2]u8{ a.id, b.id };
    try std.testing.expectEqual(c.previous[0], children[0]);
    try std.testing.expectEqual(c.previous[1], children[1]);
}

test "value buildTopo method" {
    var a = create(1);
    var b = create(2);
    b.rename("b");
    var c = a.add(&b);

    c.backward() catch {};
}
