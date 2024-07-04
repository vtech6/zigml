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

test "value create method" {
    value.resetState();
    const a = create(666, allocator);
    try expectEqual(a.value, 666);
    try expectEqual(a.label, "value");
    try expectEqual(a.op, value.OPS.init);
}

test "value add method" {
    value.resetState();
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
    value.resetState();
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
    var b = Value.create(2, allocator);
    var c = Value.create(3, allocator);
    var d = Value.create(5, allocator);
    var e = a.add(b);
    var g = c.multiply(d);
    g.rename("x");
    var f = e.add(g);
    f.setGradient(10);

    Value.prepareBackpropagation(f);
    Value.backpropagate();
    std.debug.print("To Backpropagate: {any}\n", .{value.backpropagationOrder.items});
    std.debug.print("VALUE MAP KEYS: {any}\n", .{value.valueMap.keys()});

    var _f = value.valueMap.get(f.id).?;
    var _g = value.valueMap.get(g.id).?;
    var _e = value.valueMap.get(e.id).?;
    const _d = value.valueMap.get(d.id).?;
    const _c = value.valueMap.get(c.id).?;
    //TODO: gradient of _c should equal 50. It does for _c that's a child of _g, but doesn't
    //for _c that's directly embedded in the valueMap.
    const _b = value.valueMap.get(b.id).?;
    const _a = value.valueMap.get(a.id).?;
    std.debug.print("F value: {d}, op: {any}, grad: {d}\n", .{ _f.value, _f.op, _f.gradient });
    std.debug.print("G value: {d}, op: {any}, grad: {d}\n", .{ _g.value, _g.op, _g.gradient });
    std.debug.print("E value: {d}, op: {any}, grad: {d}\n", .{ _e.value, _e.op, _e.gradient });
    std.debug.print("D value: {d}, op: {any}, grad: {d}\n", .{ _d.value, _d.op, _d.gradient });
    std.debug.print("C value: {d}, op: {any}, grad: {d}\n", .{ _c.value, _c.op, _f.gradient });
    std.debug.print("B value: {d}, op: {any}, grad: {d}\n", .{ _b.value, _b.op, _f.gradient });
    std.debug.print("A value: {d}, op: {any}, grad: {d}\n", .{ _a.value, _a.op, _a.gradient });
    std.debug.print("G children: {d}, \n", .{_g.children.items[0].gradient});
    a.deinit() catch {};
    b.deinit() catch {};
    c.deinit() catch {};
    d.deinit() catch {};
    _f.deinit() catch {};
    _e.deinit() catch {};
    _g.deinit() catch {};
}

test "value rename method" {
    var a = create(0, allocator);
    a.rename("a");
    try expectEqual(a.label, "a");
}
