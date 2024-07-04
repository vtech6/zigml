const value = @import("value.zig");
const Value = value.Value;
const std = @import("std");
const expectEqual = std.testing.expectEqual;
const allocator = std.testing.allocator;
const create = Value.create;

test "value indexing" {
    var a = create(666, allocator);
    const b = create(420, allocator);
    var c = a.add(b);
    try expectEqual(true, a.id < b.id);
    try expectEqual(true, b.id < c.id);
    c.deinit() catch {};
}

test "value create method" {
    const a = create(666, allocator);
    try expectEqual(a.value, 666);
    try expectEqual(a.label, "value");
    try expectEqual(a.op, value.OPS.init);
}

test "value add method" {
    var a = create(666, allocator);
    const b = create(420, allocator);
    var c = a.add(b);
    try expectEqual(c.value, 666 + 420);
    try expectEqual(c.children.items[0], a);
    try expectEqual(c.children.items[1], b);
    c.backward() catch {};
    try expectEqual(c.children.items[0].value, a.value);
    try expectEqual(c.children.items[1].value, b.value);
    try expectEqual(c.children.items[0].gradient, c.gradient);
    try expectEqual(c.children.items[1].gradient, c.gradient);
    try expectEqual(c.op, value.OPS.add);
    c.deinit() catch {};
}

test "value multiply method" {
    var a = create(2, allocator);
    const b = create(3, allocator);
    var c = a.multiply(b);
    try expectEqual(c.value, 6);
    try expectEqual(c.children.items[0], a);
    try expectEqual(c.children.items[1], b);
    try expectEqual(c.op, value.OPS.multiply);
    c.setGradient(10);
    try c.backward();
    try expectEqual(c.children.items[0].gradient, b.value * c.gradient);
    try expectEqual(c.children.items[1].gradient, a.value * c.gradient);
    c.deinit() catch {};
}

test "value hash map" {
    value.resetState();
    const a = create(5, allocator);
    const _a = value.valueMap.getPtr(a.id);
    try expectEqual(_a.?.value, a.value);
}

test "value backpropagate" {
    value.resetState();
    var a = Value.create(1, allocator);
    const b = Value.create(2, allocator);
    const c = Value.create(2, allocator);
    const d = Value.create(2, allocator);
    var e = a.add(b);
    var g = c.multiply(d);
    var f = e.add(g);

    Value.prepareBackpropagation(f);
    Value.backpropagate();
    std.debug.print("To Backpropagate: {any}\n", .{value.backpropagationOrder.items});

    g.deinit() catch {};
    e.deinit() catch {};
    f.deinit() catch {};
}

test "value rename method" {
    var a = create(0, allocator);
    a.rename("a");
    try expectEqual(a.label, "a");
}
