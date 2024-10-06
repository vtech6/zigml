const std = @import("std");
const value = @import("value.zig");
const Value = value.Value;
const layer = @import("layer.zig");

const Layer = layer.Layer;
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
pub const deepLayersLength: usize = 2;

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
    batchSize: usize = 4,
    steps: usize = 1,
    epochs: usize = 100,
    lossFunction: Loss = Loss.MSE,
    lossId: usize,
    momentum: f32 = 1,
    learningRate: f32 = 0.001,

    pub fn deinit(self: *Network) void {
        self.trainData.deinit();
        self.testData.deinit();
        self.layers.deinit();
    }

    pub fn create(
        inputShape: usize,
        deepLayers: [deepLayersLength]usize,
        outputShape: usize,
        trainData: ArrayList(f32),
        testData: ArrayList(f32),
        allocator: Allocator,
    ) Network {
        var layers = ArrayList(usize).init(allocator);
        const inputLayer = Layer.createLayer(inputShape, inputShape, allocator);
        layers.append(inputLayer.id) catch {};
        for (0..deepLayers.len) |layerIndex| {
            if (layerIndex == 0) {
                const newLayer = Layer.createLayer(deepLayers[layerIndex], inputShape, allocator);
                layers.append(newLayer.id) catch {};
            } else {
                const newLayer = Layer.createLayer(deepLayers[layerIndex], deepLayers[layerIndex - 1], allocator);
                layers.append(newLayer.id) catch {};
            }
        }
        const outputLayer = Layer.createLayer(outputShape, deepLayers[deepLayers.len - 1], allocator);
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

    pub fn resetLoss(self: *Network) void {
        var loss = value.valueMap.get(self.lossId).?;
        loss.value = 0.0;
        loss.update();
    }

    pub fn calculateOutput(self: *Network, input: [3]f32, y: f32) void {
        var x = ArrayList(f32).init(self.allocator);
        for (0..input.len) |inputIndex| {
            x.append(input[inputIndex]) catch {};
        }
        for (self.layers.items) |_layer| {
            if (_layer == 0) {
                var currentLayer = layer.layerMap.get(_layer).?;
                currentLayer.activateInputLayer(x);
            } else {
                var currentLayer = layer.layerMap.get(_layer).?;
                const previousLayer = layer.layerMap.get(_layer - 1).?;
                currentLayer.activateDeepLayer(previousLayer.output);
            }
        }
        x.deinit();
        switch (self.lossFunction) {
            Loss.MSE => self.meanSquaredError(y),
        }
    }

    pub fn iterateBatchItems() void {}

    pub fn iterateSteps(self: *Network) void {
        for (0..self.steps) |_| {
            iterateBatchItems();
        }
    }

    pub fn iterateBatches(self: *Network) void {
        const nBatches = @divFloor(xs.len, self.batchSize);

        self.resetLoss();
        for (0..nBatches) |batchIndex| {
            for (0..self.steps) |_| {
                self.resetLoss();
                resetGradients();
                for (0..self.batchSize) |batchItemIndex| {
                    const itemIndex = batchIndex * self.batchSize + batchItemIndex;
                    self.calculateOutput(xs[itemIndex], ys[itemIndex]);
                    self.backpropagate();
                    self.adjustValues();
                }
            }
        }
    }

    pub fn iterateEpochs(self: *Network) void {
        for (0..self.epochs) |epoch| {
            std.debug.print("epoch: {d}\n", .{epoch + 1});
            self.iterateBatches();
            const loss = value.valueMap.get(self.lossId).?;
            std.debug.print("epoch loss: {d}\n", .{loss.value});
        }
    }

    pub fn forwardPass(
        self: *Network,
    ) void {
        self.iterateEpochs();
    }

    pub fn meanSquaredError(self: *Network, yValue: f32) void {
        var loss = value.valueMap.get(self.lossId).?;
        const lastLayer = layer.layerMap.get(self.layers.items[self.layers.items.len - 1]).?;
        const targetValue = value.Value.create(yValue, self.allocator);
        const lastLayerOutput = value.valueMap.get(lastLayer.output.items[0]).?;
        const negativeOutput = lastLayerOutput.multiply(value.Value.create(-1.0, self.allocator));
        const yDifference = negativeOutput.add(targetValue);
        const yDifferenceSquared = yDifference.pow(2);
        loss.children.clearAndFree();
        loss.children.append(yDifferenceSquared.id) catch {};
        loss.value += yDifferenceSquared.value;
        loss.gradient = 1.0;
        loss.op = value.OPS.add;
        loss.update();
        std.debug.print("v: {d}, t: {d} \n", .{ lastLayerOutput.value, yValue });
    }

    pub fn backpropagate(self: *Network) void {
        var loss = value.valueMap.get(self.lossId).?;
        loss.backpropagate();
    }

    pub fn adjustValues(self: *Network) void {
        const valueMap = value.valueMap;
        const valueMapKeys = valueMap.keys();
        for (valueMapKeys) |valueKey| {
            var _value = valueMap.get(valueKey).?;
            _value.value += (_value.gradient * (-self.learningRate) * self.momentum);
            _value.update();
        }
    }

    pub fn resetGradients() void {
        const valueMap = value.valueMap;
        const valueMapKeys = valueMap.keys();
        for (valueMapKeys) |valueKey| {
            var _value = valueMap.get(valueKey).?;
            _value.resetGradient();
        }
    }

    pub fn printNeurons(self: Network) void {
        for (self.layers.items) |layerId| {
            const layerValue = layer.layerMap.get(layerId).?;
            layerValue.printNeurons();
        }
    }
};
