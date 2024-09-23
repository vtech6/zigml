const value = @import("value.zig");
const Value = value.Value;
const std = @import("std");
const expectEqual = std.testing.expectEqual;
const allocator = std.testing.allocator;
const create = Value.create;

test "value indexing" {
    value.resetState();
    var a = create(666, allocator);
    const b = create(420, allocator);
    var c = a.add(b);
    try expectEqual(true, a.id < b.id);
    try expectEqual(true, b.id < c.id);
    c.deinit() catch {};
}

test "value add mathod" {
    value.resetState();
    var a = create(21, allocator);
    const b = create(37, allocator);
    var c = a.add(b);
    try expectEqual(58, c.value);
    c.deinit() catch {};
}

test "value add mathod extensive" {
    value.resetState();
    const a = create(21, allocator);
    const b = create(37, allocator);
    const c = a.add(b);
    const d = a.add(c);
    const e = c.add(a);
    const f = e.add(d);
    try expectEqual(158, f.value);
    value.cleanup();
}

test "value update method" {
    value.resetState();
    var a = create(666, allocator);
    a.value = 999;
    a.update();
    const _a = value.valueMap.get(a.id).?;
    try expectEqual(999, _a.value);
}
