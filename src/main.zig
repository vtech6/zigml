const std = @import("std");
const value = @import("./model/value.zig");
const Value = value.Value;
const neuron = @import("./model/neuron.zig");
const Neuron = neuron.Neuron;
pub const testAllocator = std.heap.page_allocator;
pub fn main() !void {
    var _neuron = Neuron.create(2, testAllocator);
    var input = std.ArrayList(usize).init(testAllocator);
    try input.append(Value.create(3, testAllocator).id);
    try input.append(Value.create(5, testAllocator).id);
    try _neuron.activateDeep(input);
    const w1 = value.valueMap.get(_neuron.weights.items[0]).?.value;
    const w2 = value.valueMap.get(_neuron.weights.items[1]).?.value;
    const neuronActivation = value.valueMap.get(_neuron.activation).?.value;
    var inputItem1 = value.valueMap.get(input.items[0]).?;
    var inputItem2 = value.valueMap.get(input.items[1]).?;
    std.debug.print("i1, i2, w1, w2: {d}, {d}, {d}, {d}\n", .{
        inputItem1.value,
        inputItem2.value,
        w1,
        w2,
    });
    std.debug.print("Neuron activation: {d}\n", .{neuronActivation});
    var neuronActivationValue = value.valueMap.get(_neuron.activation).?;
    neuronActivationValue.gradient = 1.0;
    neuronActivationValue.backpropagate();
    neuronActivationValue.visualize();
    inputItem1 = value.valueMap.get(input.items[0]).?;
    inputItem2 = value.valueMap.get(input.items[1]).?;

    std.debug.print("i1, i2: {d}, {d}\n", .{
        inputItem1.gradient,
        inputItem2.gradient,
    });

    input.deinit();
    neuron.cleanup();
}

test "simple test" {
    var list = std.ArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}
