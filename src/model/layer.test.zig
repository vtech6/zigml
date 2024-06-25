const std = @import("std");
const print = std.debug.print;
const neuron = @import("neuron.zig");
const Neuron = neuron.Neuron;
const testAllocator = std.testing.allocator;

test "create Neuron" {
    var n1 = Neuron.create(testAllocator, 2);
    print("weights: {any}\n", .{n1});
    print("weights: {}\n", .{n1.weights.getLast()});
    try n1.freeMemory();
}
