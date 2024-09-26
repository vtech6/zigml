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
    c.deinit();
}

test "value add method" {
    value.resetState();
    var a = create(21, allocator);
    const b = create(37, allocator);
    var c = a.add(b);
    try expectEqual(58, c.value);
    c.deinit();
}

test "value add method extensive" {
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

test "value multiply method extensive" {
    value.resetState();
    const a = create(21, allocator);
    const b = create(37, allocator);
    const c = a.multiply(b);
    const d = a.multiply(c);
    const e = c.multiply(a);
    const f = e.multiply(d);
    try expectEqual(266244500, f.value);
    value.cleanup();
}

test "value tanh method" {
    value.resetState();
    const a = create(0.5, allocator);
    const b = a.tanh();
    try expectEqual(0.46211714, b.value);
    value.cleanup();
}

test "backpropagation" {
    value.resetState();
    const a = create(0.7, allocator);
    const b = create(-1, allocator);
    const c = create(0.2, allocator);
    const d = create(1, allocator);
    const e = create(-0.3, allocator);
    const ab = a.multiply(b);
    const cd = c.multiply(d);
    const abcd = ab.add(cd);
    const abcde = abcd.add(e);
    var abcdef = abcde.tanh();
    abcdef.gradient = 1.0;
    abcdef.backpropagate();
    const _a = value.valueMap.get(a.id).?;
    const _b = value.valueMap.get(b.id).?;
    const _c = value.valueMap.get(c.id).?;
    const _d = value.valueMap.get(d.id).?;
    const _e = value.valueMap.get(e.id).?;
    const _ab = value.valueMap.get(ab.id).?;
    const _cd = value.valueMap.get(cd.id).?;
    const _abcd = value.valueMap.get(abcd.id).?;
    const _abcde = value.valueMap.get(abcde.id).?;
    std.debug.print("a: {d},\nb: {d},\nc: {d},\nd: {d},\ne: {d},\nab: {d},\ncd: {d},\nabcd: {d},\nabcde: {d},\n", .{
        _a.gradient,
        _b.gradient,
        _c.gradient,
        _d.gradient,
        _e.gradient,
        _ab.gradient,
        _cd.gradient,
        _abcd.gradient,
        _abcde.gradient,
    });
    try expectEqual(_a.gradient == 0, false);
    try expectEqual(_b.gradient == 0, false);
    try expectEqual(_c.gradient == 0, false);
    try expectEqual(_d.gradient == 0, false);
    try expectEqual(_e.gradient == 0, false);
    try expectEqual(_ab.gradient == 0, false);
    try expectEqual(_cd.gradient == 0, false);
    try expectEqual(_abcd.gradient == 0, false);
    try expectEqual(_abcde.gradient == 0, false);
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
