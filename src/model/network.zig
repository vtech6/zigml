const std = @import("std");
const value = @import("value.zig");
const Value = value.Value;
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
    batchSize: usize = 1,
    steps: usize = 5,
    epochs: i32 = 10,
    lossFunction: Loss = Loss.MSE,
    lossId: usize,
    momentum: f32 = 1,
    learningRate: f32 = 0.1,

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
        const loss = value.Value.create(0.0, allocator);
        return Network{
            .layers = layers,
            .trainData = trainData,
            .testData = testData,
            .allocator = allocator,
            .lossId = loss.id,
        };
    }

    pub fn iterateStep(self: *Network, input: [3]f32, y: f32) void {
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
            Loss.MSE => self.meanSquaredError(y),
        }
    }

    pub fn forwardPass(
        self: *Network,
    ) void {
        self.iterateBatches();
        const loss = value.valueMap.get(self.lossId).?;
        std.debug.print("PRINTING LOSS {d}\n", .{loss.value});
    }

    pub fn iterateBatches(self: *Network) void {
        var loop: usize = 0;
        const nBatches = @divFloor(xs.len, self.batchSize);
        for (0..nBatches) |batchIndex| {
            for (0..self.batchSize) |batchItemIndex| {
                const itemIndex = batchIndex * self.batchSize + batchItemIndex;
                for (0..self.steps) |step| {
                    loop += 1;
                    std.debug.print("iteration {d}, step {d}, batch {d}, ", .{ loop, step + 1, batchIndex + 1 });
                    self.iterateStep(xs[itemIndex], ys[itemIndex]);
                    self.backpropagate();
                    self.adjustValues();
                }
            }
        }
    }

    pub fn backpropagate(self: *Network) void {
        var loss = value.valueMap.get(self.lossId).?;
        loss.gradient = 1.0;
        loss.backpropagate();
        loss.update();
    }

    pub fn adjustValues(self: *Network) void {
        const valueMap = value.valueMap;
        const valueMapKeys = valueMap.keys();
        for (valueMapKeys) |valueKey| {
            var _value = valueMap.get(valueKey).?;
            if (_value.gradient != 0.0) {
                _value.value += (_value.gradient * (-self.learningRate) * self.momentum);
                _value.update();
            }
        }
    }

    pub fn meanSquaredError(self: *Network, yValue: f32) void {
        var loss = value.valueMap.get(self.lossId).?;
        const lastLayer = layer.layerMap.get(self.layers.items[self.layers.items.len - 1]).?;
        const targetValue = value.Value.create(yValue, self.allocator);
        const lastLayerOutput = value.valueMap.get(lastLayer.output.items[0]).?;
        const negativeOutput = lastLayerOutput.multiply(value.Value.create(-1.0, self.allocator));
        const yDifference = negativeOutput.add(targetValue);
        loss.value += yDifference.value;
        loss.children.append(yDifference.id) catch {};
        loss.update();
        std.debug.print("loss: {d}, value: {d}, traget: {d} \n", .{ loss.value, lastLayerOutput.value, yValue });
    }
};
