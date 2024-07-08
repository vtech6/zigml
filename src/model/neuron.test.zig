const std = @import("std");
const print = std.debug.print;
const neuron = @import("neuron.zig");
const Neuron = neuron.Neuron;
const testAllocator = std.testing.allocator;
const expectEqual = std.testing.expectEqual;

test "create Neuron" {
    var _neuron = Neuron.create(2, testAllocator);
    try expectEqual(2, _neuron.weights.items.len);
    try _neuron.freeMemory();
}

test "activate Neuron" {
    var _neuron = Neuron.create(2, testAllocator);
    var input = std.ArrayList(f32).init(testAllocator);
    try input.append(0.0);
    try input.append(1.1);
    try _neuron.activateInput(input);
    const w2 = _neuron.weights.items[1].value;
    try expectEqual(_neuron.output.items[1].value, (w2 * input.items[1]) + _neuron.bias.value);
    try _neuron.freeMemory();
    input.deinit();
}
