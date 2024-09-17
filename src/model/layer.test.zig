const std = @import("std");
const print = std.debug.print;
const layer = @import("layer.zig");
const Layer = layer.Layer;
const testAllocator = std.testing.allocator;
const value = @import("value.zig");
const Value = value.Value;
const expectEqual = std.testing.expectEqual;
test "create input layer" {
    var input = std.ArrayList(f32).init(testAllocator);
    try input.append(1.0);
    try input.append(3.0);
    var l1 = Layer.createInputLayer(input.items.len, 4, testAllocator);
    print("neurons: {any}\n", .{l1.neurons.items.len});
    print("output: {any}\n", .{l1.output.items.len});
    try expectEqual(l1.neurons.items.len, 4);
    try expectEqual(l1.output.items.len, 4);
    try expectEqual(l1.output.items[0].value, 0);

    input.deinit();
    try l1.deinit();
}
test "activate input layer" {
    var input = std.ArrayList(f32).init(testAllocator);
    try input.append(1.0);
    try input.append(3.0);
    var l1 = Layer.createInputLayer(input.items.len, 4, testAllocator);
    try l1.activateInputLayer(input);
    print("neurons: {any}\n", .{l1.neurons.items.len});
    print("output: {any}\n", .{l1.output.items.len});
    print("first output: {d}\n", .{l1.output.items[0].value});
    try expectEqual(l1.output.items[0].value == 0, false);

    input.deinit();
    try l1.deinit();
}

test "activate deep layer" {
    var input = std.ArrayList(Value).init(testAllocator);
    try input.append(Value.create(1, testAllocator));
    try input.append(Value.create(2, testAllocator));
    var l1 = Layer.createDeepLayer(input.items.len, 3, testAllocator);
    try l1.activateDeepLayer(input);
    print("deep neurons: {any}\n", .{l1.neurons.items.len});
    print("deep output: {any}\n", .{l1.output.items.len});
    print("first deep output: {d}\n", .{l1.output.items[0].value});
    try expectEqual(l1.output.items[0].value == 0, false);

    input.deinit();
    try l1.deinit();
}

test "combine layers" {
    value.resetState();
    var input = std.ArrayList(f32).init(testAllocator);
    try input.append(1.0);
    try input.append(3.0);
    var l1 = Layer.createInputLayer(input.items.len, 3, testAllocator);
    try l1.activateInputLayer(input);
    var l2 = Layer.createDeepLayer(l1.output.items.len, 1, testAllocator);
    print("l2 output op pre-backprop: {any}\n", .{l2.output.items[0].op});
    try l2.activateDeepLayer(l1.output);
    print("l1 neurons: {any}\n", .{l1.neurons.items.len});
    print("l1 first output: {d}\n", .{l1.output.items[0].value});
    print("l2 neurons: {any}\n", .{l2.neurons.items.len});
    print("l2 output: {d}\n", .{l2.output.items[0].value});
    print("l2 output children len: {d}\n", .{l2.output.items[0].children.items.len});
    const l1n1w1 = value.valueMap.getPtr(l1.neurons.items[0].weights.items[0].id).?;
    const l1n1w1grad = l1n1w1.gradient;
    print("l1 n1 w1 grad: {d}\n", .{l1n1w1.gradient});
    try expectEqual(l2.output.items[0].value == 0, false);
    try expectEqual(l1n1w1grad == 1, true);
    var finalOutput = l2.output.items[0];
    finalOutput.backward() catch {};
    print("l2 output op child1: {any}\n", .{l2.output.items[0].children.items[1]});

    Value.backpropagate();
    print("l2 output op post-backprop: {any}\n", .{l2.output.items[0].op});
    print("l1n1 output op {any}\n", .{l1.neurons.items[0].weights.items[0].gradient});

    print("l1 n1 w1 grad: {d}\n", .{l1n1w1grad});
    input.deinit();
    //TODO: FIX BACKPROP SO THAT L1N1W1 gradient is not equal 1;
}
