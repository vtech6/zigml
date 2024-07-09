const std = @import("std");
const print = std.debug.print;
const layer = @import("layer.zig");
const Layer = layer.Layer;
const testAllocator = std.testing.allocator;
const value = @import("value.zig");
const expectEqual = std.testing.expectEqual;
test "create input layer" {
    var input = std.ArrayList(f32).init(testAllocator);
    try input.append(1.0);
    try input.append(3.0);
    var l1 = Layer.createInputLayer(input, 4, testAllocator);
    print("neurons: {any}\n", .{l1.neurons.items.len});
    print("output: {any}\n", .{l1.output.items.len});

    input.deinit();
    try l1.deinit();
}
test "activate input layer" {
    var input = std.ArrayList(f32).init(testAllocator);
    try input.append(1.0);
    try input.append(3.0);
    var l1 = Layer.createInputLayer(input, 4, testAllocator);
    try l1.activateInputLayer(input);
    print("neurons: {any}\n", .{l1.neurons.items.len});
    print("output: {any}\n", .{l1.output.items.len});
    print("first output: {d}\n", .{l1.output.items[0].value});

    input.deinit();
    try l1.deinit();
}
