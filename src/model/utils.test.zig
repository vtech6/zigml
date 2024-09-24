//xs = [
// [2.0, 3.0, -1.0],
// [3.0, -1.0, 0.5],
// [0.5, 1.0, 1.0],
// [1.0, 1.0, -1.0],
//]
//ys = [1.0, -1.0, -1.0, 1.0]
const std = @import("std");
const ArrayList = std.ArrayList;
const network = @import("network.zig");
const Network = network.Network;
const testing = std.testing;
const layer = @import("layer.zig");
test "xd" {
    const trainData = std.ArrayList(f32).init(testing.allocator);
    var newNetwork = Network.create(3, 2, 1, trainData, trainData, testing.allocator);
    newNetwork.forwardPass();
    trainData.deinit();
    layer.cleanup();
    newNetwork.deinit();
}
