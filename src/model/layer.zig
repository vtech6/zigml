const std = @import("std");
const value = @import("value.zig");
const neuron = @import("neuron.zig");
const Neuron = neuron.Neuron;
const Value = value.Value;
const Allocator = std.mem.Allocator;
pub const Layer = struct {
    output: std.ArrayList(Value),
    neurons: std.ArrayList(Neuron),

    pub fn freeMemory(self: *Layer) !void {
        self.output.deinit();
        self.neurons.deinit();
    }

    pub fn createInputLayer(allocator: Allocator, inputShape: usize) Layer {
        var neurons = std.ArrayList(Neuron).init(allocator);
        for (0..inputShape) |_| {
            const newNeuron = Neuron.create(allocator, inputShape);
            neurons.append(newNeuron);
        }
    }

    pub fn feedForwardInputLayer() !void {}

    pub fn createDeepLayer(
        allocator: Allocator,
        previousLayerShape: usize,
    ) Layer {
        var neurons = std.ArrayList(Neuron).init(allocator);
        for (0..previousLayerShape) |_| {
            const newNeuron = Neuron.create(allocator, previousLayerShape);
            neurons.append(newNeuron);
        }
    }
    pub fn createOutputLayer(
        allocator: Allocator,
        previousLayerShape: usize,
    ) Layer {
        var neurons = std.ArrayList(Neuron).init(allocator);
        for (0..previousLayerShape) |_| {
            const newNeuron = Neuron.create(allocator, previousLayerShape);
            neurons.append(newNeuron);
        }
    }

    pub fn create(allocator: std.mem.Allocator, inputShape: usize) Neuron {
        var weights = std.ArrayList(Value).init(allocator);
        for (0..inputShape) |_| {
            const newValue = Value.create(10);
            weights.append(newValue) catch {};
        }
        const newNeuron = Neuron{
            .weights = weights,
            .bias = 0,
        };
        return newNeuron;
    }
};
