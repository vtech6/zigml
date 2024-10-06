const std = @import("std");
const ArrayList = std.ArrayList;
const network = @import("network.zig");
const Network = network.Network;
const testing = std.testing;
const layer = @import("layer.zig");
const expectEqual = testing.expectEqual;
pub const xs = [4][3]f32{
    .{ 2.0, 3.0, -1.0 },
    .{ 3.0, -1.0, 0.5 },
    .{ 0.5, 1.0, 1.0 },
    .{ 1.0, 1.0, -1.0 },
};

pub const ys = [4]f32{ 1.0, -1.0, -1.0, 1.0 };

test "Network create method" {
    const trainData = std.ArrayList(f32).init(testing.allocator);
    const deepLayers = [2]usize{ 3, 4 };
    var newNetwork = Network.create(
        3,
        deepLayers,
        1,
        trainData,
        trainData,
        testing.allocator,
    );
    try expectEqual(newNetwork.layers.items.len, 4);
    trainData.deinit();
    layer.cleanup();
    newNetwork.deinit();
}

test "Network forward pass" {
    const trainData = std.ArrayList(f32).init(testing.allocator);
    const deepLayers = [2]usize{ 3, 3 };
    var newNetwork = Network.create(
        3,
        deepLayers,
        1,
        trainData,
        trainData,
        testing.allocator,
    );
    newNetwork.forwardPass();
    try expectEqual(newNetwork.layers.items.len, 4);
    trainData.deinit();
    layer.cleanup();
    newNetwork.deinit();
}
