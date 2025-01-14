const std = @import("std");
const math = std.math;
const print = std.debug.print;
const neuron = @import("neuron.zig");
const Neuron = neuron.Neuron;
const testAllocator = std.testing.allocator;
const expectEqual = std.testing.expectEqual;
const value = @import("value.zig");
const Value = value.Value;
const Activation = @import("activation.zig").Activation;

test "create Neuron" {
    const _neuron = Neuron.create(2, testAllocator);
    try expectEqual(2, _neuron.weights.items.len);
    neuron.cleanup();
}

test "activate input Neuron" {
    var _neuron = Neuron.create(2, testAllocator);
    var input = std.ArrayList(f32).init(testAllocator);
    try input.append(0.0);
    try input.append(1.1);
    try _neuron.activateInput(
        input,
    );
    const w1 = value.valueMap.get(_neuron.weights.items[0]).?.value;
    const w2 = value.valueMap.get(_neuron.weights.items[1]).?.value;
    const bias = value.valueMap.get(_neuron.bias).?.value;
    const neuronActivation = value.valueMap.get(_neuron.activation).?.value;
    const manualActivation = ((w1 * input.items[0]) + (w2 * input.items[1]) + bias);
    const manualActivationTanh = (math.exp(2 * manualActivation) - 1) / (math.exp(2 * manualActivation) + 1);
    try expectEqual(manualActivationTanh, neuronActivation);
    input.deinit();
    neuron.cleanup();
}

test "activate deep Neuron" {
    var _neuron = Neuron.create(2, testAllocator);
    var input = std.ArrayList(usize).init(testAllocator);
    try input.append(Value.create(0.5, testAllocator).id);
    try input.append(Value.create(-1, testAllocator).id);
    try _neuron.activateDeep(input);
    const w1 = value.valueMap.get(_neuron.weights.items[0]).?.value;
    const w2 = value.valueMap.get(_neuron.weights.items[1]).?.value;
    const bias = value.valueMap.get(_neuron.bias).?.value;
    const w2Activation = value.valueMap.get(_neuron.output.items[1]).?.value;
    const neuronActivation = value.valueMap.get(_neuron.activation).?.value;
    var inputItem1 = value.valueMap.get(input.items[0]).?;
    var inputItem2 = value.valueMap.get(input.items[1]).?;
    const manualActivation = ((w1 * inputItem1.value) + (w2 * inputItem2.value) + bias);
    const manualActivationTanh = (math.exp(2 * manualActivation) - 1) / (math.exp(2 * manualActivation) + 1);
    try expectEqual(w2Activation, (w2 * inputItem2.value));
    try expectEqual(manualActivationTanh, neuronActivation);
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
    inputItem1 = value.valueMap.get(input.items[0]).?;
    inputItem2 = value.valueMap.get(input.items[1]).?;

    try expectEqual(inputItem1.gradient == 0, false);
    try expectEqual(inputItem2.gradient == 0, false);

    input.deinit();
    neuron.cleanup();
}

test "activate deep Neuron multiple times" {
    var input = std.ArrayList(f32).init(testAllocator);
    try input.append(0.0);
    try input.append(1.1);
    var _neuron = Neuron.create(2, testAllocator);
    var _neuron2 = Neuron.create(2, testAllocator);
    var _neuron3 = Neuron.create(2, testAllocator);
    for (0..3) |_| {
        try _neuron.activateInput(
            input,
        );
        var _previousActivations = std.ArrayList(usize).init(testAllocator);
        try _neuron2.activateInput(
            input,
        );

        _previousActivations.append(_neuron.activation) catch {};
        _previousActivations.append(_neuron2.activation) catch {};

        try _neuron3.activateDeep(_previousActivations);

        _previousActivations.deinit();
    }

    input.deinit();
    neuron.cleanup();
}
