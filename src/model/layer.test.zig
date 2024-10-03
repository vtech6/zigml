const std = @import("std");
const value = @import("value.zig");
const neuron = @import("neuron.zig");
const layer = @import("layer.zig");
const Neuron = neuron.Neuron;
const Value = value.Value;
const Layer = layer.Layer;
const allocator = std.testing.allocator;
const expectEqual = std.testing.expectEqual;

test "create input layer" {
    const inputLayer = Layer.createLayer(3, allocator);
    try expectEqual(3, inputLayer.neurons.items.len);
    try expectEqual(3, inputLayer.output.items.len);
    layer.cleanup();
}

test "activate input layer" {
    layer.resetState();
    var newLayer = Layer.createLayer(2, allocator);
    var input = std.ArrayList(f32).init(allocator);
    try input.append(0.0);
    try input.append(1.1);
    newLayer.activateInputLayer(input);
    const output1 = value.valueMap.get(newLayer.output.items[0]).?;
    const output2 = value.valueMap.get(newLayer.output.items[1]).?;

    try expectEqual(output1.value, 0.4812011);
    try expectEqual(output2.value, 0.2508039);
    layer.cleanup();
    input.deinit();
}

test "activate deep layer" {
    layer.resetState();
    var newLayer = Layer.createLayer(2, allocator);
    var input = std.ArrayList(usize).init(allocator);
    try input.append(Value.create(0.5, allocator).id);
    try input.append(Value.create(-1, allocator).id);
    newLayer.activateDeepLayer(input);
    const output1 = value.valueMap.get(newLayer.output.items[0]).?;
    const output2 = value.valueMap.get(newLayer.output.items[1]).?;

    try expectEqual(output1.value, 0.21914873);
    try expectEqual(output2.value, -0.0615547);
    layer.cleanup();
    input.deinit();
}

test "pass value between layers" {
    layer.resetState();
    var newLayer = Layer.createLayer(2, allocator);
    var input = std.ArrayList(f32).init(allocator);
    try input.append(0.0);
    try input.append(1.1);
    newLayer.activateInputLayer(input);
    const layer1 = layer.layerMap.get(newLayer.id).?;
    var newLayer2 = Layer.createLayer(2, allocator);
    newLayer2.activateDeepLayer(layer1.output);
    const output1 = value.valueMap.get(newLayer.output.items[0]).?;
    const output2 = value.valueMap.get(newLayer.output.items[1]).?;
    const output3 = value.valueMap.get(newLayer2.output.items[0]).?;
    const output4 = value.valueMap.get(newLayer2.output.items[1]).?;

    try expectEqual(output1.value, 0.4840346);
    try expectEqual(output2.value, 0.9437493);
    try expectEqual(output3.value, 0.87674046);
    try expectEqual(output4.value, 0.92418176);

    layer.cleanup();
    input.deinit();
}
