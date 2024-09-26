const std = @import("std");
const ArrayList = std.ArrayList;
const network = @import("network.zig");
const Network = network.Network;
const testing = std.testing;
const layer = @import("layer.zig");
test "Network create method" {
    const trainData = std.ArrayList(f32).init(testing.allocator);
    var newNetwork = Network.create(3, 2, 1, trainData, trainData, testing.allocator);
    trainData.deinit();
    layer.cleanup();
    newNetwork.deinit();
}
