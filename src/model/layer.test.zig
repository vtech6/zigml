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
    var l1 = Layer.createDeepLayer(input.items.len, 4, testAllocator);
    try l1.activateDeepLayer(input);
    print("deep neurons: {any}\n", .{l1.neurons.items.len});
    print("deep output: {any}\n", .{l1.output.items.len});
    print("first deep output: {d}\n", .{l1.output.items[0].value});
    try expectEqual(l1.output.items[0].value == 0, false);

    input.deinit();
    try l1.deinit();
}
