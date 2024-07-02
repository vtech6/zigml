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

test "feedForward Neuron" {
    var _neuron = Neuron.create(2, testAllocator);
    const input = [2]f32{ 0.0, 1.0 };
    try _neuron.activateInput(input, testAllocator);
    try expectEqual(_neuron.output.items[1].value, 10.0);
    try _neuron.freeMemory();
}
