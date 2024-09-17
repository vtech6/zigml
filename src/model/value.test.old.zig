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
    const b = Value.create(2, allocator);
    var c = Value.create(3, allocator);
    const d = Value.create(5, allocator);
    var e = a.add(b);
    var g = c.multiply(d);
    var f = e.add(g);
    g.rename("g");
    f.setGradient(10);

    Value.prepareBackpropagation(f);
    Value.backpropagate();

    const backpropagationOrder = [7]usize{ 6, 4, 0, 1, 5, 2, 3 };
    for (0..backpropagationOrder.len) |index| {
        try expectEqual(value.backpropagationOrder.items[index], backpropagationOrder[index]);
    }

    var _f = value.valueMap.get(f.id).?;
    var _g = value.valueMap.get(g.id).?;
    var _e = value.valueMap.get(e.id).?;
    const _d = value.valueMap.get(d.id).?;
    const _c = value.valueMap.get(c.id).?;
    const _b = value.valueMap.get(b.id).?;
    const _a = value.valueMap.get(a.id).?;

    try expectEqual(_f.value, 18);
    try expectEqual(_f.gradient, 10);
    try expectEqual(_e.value, 3);
    try expectEqual(_e.gradient, 10);
    try expectEqual(_g.value, 15);
    try expectEqual(_g.gradient, 10);
    try expectEqual(_d.value, 5);
    try expectEqual(_d.gradient, 30);
    try expectEqual(_c.value, 3);
    try expectEqual(_c.gradient, 50);
    try expectEqual(_b.value, 2);
    try expectEqual(_b.gradient, 10);
    try expectEqual(_a.value, 1);
    try expectEqual(_a.gradient, 10);

    _f.deinit() catch {};
    _e.deinit() catch {};
    _g.deinit() catch {};
}

test "value rename method" {
    var a = create(0, allocator);
    a.rename("a");
    try expectEqual(a.label, "a");
}

test "value activate method" {
    value.resetState();
    var newChildren = std.ArrayList(Value).init(allocator);
    const a = create(1, allocator);
    const b = create(2, allocator);
    const c = create(3, allocator);
    try newChildren.append(a);
    try newChildren.append(b);
    try newChildren.append(c);
    var d = create(8, allocator);
    d.children = newChildren;
    d.op = value.OPS.activate;
    d.setGradient(666);

    try expectEqual(a.gradient, 1);
    try expectEqual(b.gradient, 1);
    try expectEqual(c.gradient, 1);
    d.backward() catch {};

    try expectEqual(value.valueMap.getPtr(a.id).?.gradient, 666);
    try expectEqual(value.valueMap.getPtr(b.id).?.gradient, 666);
    try expectEqual(value.valueMap.getPtr(c.id).?.gradient, 666);
    d.deinit() catch {};
}

test "simulate neuron activation" {
    value.resetState();
    const weight = create(1, allocator);
    const bias = create(2, allocator);
    const inputValue = create(3, allocator);
    var weighedInput = inputValue.multiply(weight);
    const activation = weighedInput.add(bias);
    Value.prepareBackpropagation(activation);
    Value.backpropagate();
    value.valueMap.getPtr(weighedInput.id).?.deinit() catch {};
    value.valueMap.getPtr(activation.id).?.deinit() catch {};
}
