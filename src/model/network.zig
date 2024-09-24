const std = @import("std");
const value = @import("value.zig");
const layer = @import("layer.zig");
const Layer = layer.Layer;
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;

pub const Loss = enum {
    MSE,
};

pub const xs = [4][3]f32{
    .{ 2.0, 3.0, -1.0 },
    .{ 3.0, -1.0, 0.5 },
    .{ 0.5, 1.0, 1.0 },
    .{ 1.0, 1.0, -1.0 },
};

pub const ys = [4]f32{ 1.0, -1.0, -1.0, 1.0 };

pub const Network = struct {
    trainData: ArrayList(f32),
    testData: ArrayList(f32),
    layers: ArrayList(usize),
    allocator: Allocator,
    steps: usize = 3,
    epochs: i32 = 5,
    lossFunction: Loss = Loss.MSE,

    pub fn deinit(self: *Network) void {
        self.trainData.deinit();
        self.testData.deinit();
        self.layers.deinit();
    }

    pub fn create(
        inputShape: usize,
        deepLayers: usize,
        outputShape: usize,
        trainData: ArrayList(f32),
        testData: ArrayList(f32),
        allocator: Allocator,
    ) Network {
        var layers = ArrayList(usize).init(allocator);
        const inputLayer = Layer.createLayer(inputShape, allocator);
        layers.append(inputLayer.id) catch {};
        for (0..deepLayers) |_layerShape| {
            const newLayer = Layer.createLayer(_layerShape, allocator);
            layers.append(newLayer.id) catch {};
        }
        const outputLayer = Layer.createLayer(outputShape, allocator);
        layers.append(outputLayer.id) catch {};
        return Network{
            .layers = layers,
            .trainData = trainData,
            .testData = testData,
            .allocator = allocator,
        };
    }

    pub fn iterateStep(self: *Network, input: [3]f32, y: f32, lossId: usize) void {
        var x = ArrayList(f32).init(self.allocator);
        for (0..input.len) |inputIndex| {
            x.append(input[inputIndex]) catch {};
        }
        for (self.layers.items, 0..) |layerValue, layerIndex| {
            if (layerIndex == 0) {
                var _layer = layer.layerMap.get(layerValue).?;
                _layer.activateInputLayer(x);
            } else {
                var currentLayer = layer.layerMap.get(layerValue).?;
                const previousLayer = layer.layerMap.get(layerIndex - 1).?;
                currentLayer.activateDeepLayer(previousLayer.output);
            }
        }
        x.deinit();
        switch (self.lossFunction) {
            Loss.MSE => {
                var loss = value.valueMap.get(lossId).?;
                const lastLayer = layer.layerMap.get(self.layers.items[self.layers.items.len - 1]).?;
                const targetValue = value.Value.create(y, self.allocator);

                const lastLayerOutput = value.valueMap.get(lastLayer.output.items[0]).?;
                const negativeOutput = lastLayerOutput.multiply(value.Value.create(-1.0, self.allocator));
                const yDifference = negativeOutput.add(targetValue);
                loss.value += yDifference.value;
                loss.children.append(yDifference.id) catch {};
                loss.update();
                std.debug.print("New Loss: {d}\n", .{loss.value});
            },
        }
    }

    pub fn forwardPass(
        self: *Network,
    ) void {
        var loss = value.Value.create(0.0, self.allocator);

        for (0..xs.len) |rowIndex| {
            for (0..self.steps) |_| {
                self.iterateStep(xs[rowIndex], ys[rowIndex], loss.id);
            }
        }
        loss = value.valueMap.get(loss.id).?;
        std.debug.print("PRINTING LOSS {d}\n", .{loss.value});
    }
};
