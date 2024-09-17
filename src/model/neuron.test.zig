const std = @import("std");
const print = std.debug.print;
const neuron = @import("neuron.zig");
const Neuron = neuron.Neuron;
const testAllocator = std.testing.allocator;
const expectEqual = std.testing.expectEqual;
const value = @import("value.zig");
const Value = value.Value;

test "create Neuron" {
    var _neuron = Neuron.create(2, testAllocator);
    try expectEqual(2, _neuron.weights.items.len);
    try _neuron.deinit();
}

test "activate input Neuron" {
    value.resetState();
    var _neuron = Neuron.create(2, testAllocator);
    var input = std.ArrayList(f32).init(testAllocator);
    try input.append(0.0);
    try input.append(1.1);
    try _neuron.activateInput(input);
    const w2 = _neuron.weights.items[1].value;
    try expectEqual(_neuron.output.items[1].value, (w2 * input.items[1]) + _neuron.bias.value);
    std.debug.print("Activation: {d}\n", .{_neuron.activation.value});
    try _neuron.deinit();

    input.deinit();
}

test "activate deep Neuron" {
    value.resetState();
    var _neuron = Neuron.create(2, testAllocator);
    var input = std.ArrayList(Value).init(testAllocator);
    const a = Value.create(1.0, testAllocator);
    try input.append(a);
    const b = Value.create(2.0, testAllocator);
    try input.append(b);

    try _neuron.activateDeep(input);
    std.debug.print("Activation: {d}\n", .{_neuron.activation.value});
    try _neuron.deinit();

    input.deinit();
}

test "backpropagate Neurons" {
    value.resetState();
    var layer = std.ArrayList(Value).init(testAllocator);

    var _neuron = Neuron.create(2, testAllocator);
    var input = std.ArrayList(f32).init(testAllocator);
    try input.append(0.0);
    try input.append(1.1);
    try _neuron.activateInput(input);
    layer.append(_neuron.activation) catch {};
    std.debug.print("Activation: {d}\n", .{_neuron.activation.value});
    var _neuron2 = Neuron.create(1, testAllocator);
    _neuron2.activateDeep(layer) catch {};
    layer.deinit();
    std.debug.print("Activation: {d}\n", .{_neuron2.activation.value});
    input.deinit();
    _neuron2.deinit() catch {};
}
