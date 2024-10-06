const std = @import("std");
const value = @import("./model/value.zig");
const Value = value.Value;
const Network = @import("./model/network.zig").Network;
const neuron = @import("./model/neuron.zig");
const Neuron = neuron.Neuron;
const layer = @import("./model/layer.zig");
pub const testAllocator = std.heap.page_allocator;
pub fn main() !void {
    const trainData = std.ArrayList(f32).init(testAllocator);
    const deepLayers = [2]usize{ 3, 3 };
    var newNetwork = Network.create(3, deepLayers, 1, trainData, trainData, testAllocator);
    newNetwork.forwardPass();
    value.valueMap.get(537).?.visualize();
    trainData.deinit();
    layer.cleanup();
    newNetwork.deinit();
}

test "simple test" {
    var list = std.ArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}
