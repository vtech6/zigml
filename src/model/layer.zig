const std = @import("std");
const value = @import("value.zig");
const neuron = @import("neuron.zig");
const Neuron = neuron.Neuron;
const Value = value.Value;
const Allocator = std.mem.Allocator;
pub const Layer = struct {
    output: std.ArrayList(Value),
    neurons: std.ArrayList(Neuron),
    allocator: Allocator,

    pub fn deinit(self: *Layer) !void {
        self.output.deinit();
        for (self.neurons.items) |item| {
            try item.deinit();
        }
        self.neurons.deinit();
    }

    pub fn createInputLayer(
        inputLen: usize,
        layerSize: usize,
        allocator: Allocator,
    ) Layer {
        var neurons = std.ArrayList(Neuron).init(allocator);
        var output = std.ArrayList(Value).init(allocator);
        for (0..layerSize) |_| {
            const newNeuron = Neuron.create(inputLen, allocator);
            output.append(newNeuron.activation) catch {};
            neurons.append(newNeuron) catch {};
        }
        const newLayer = Layer{
            .neurons = neurons,
            .output = output,
            .allocator = allocator,
        };
        return newLayer;
    }

    pub fn activateInputLayer(self: *Layer, input: std.ArrayList(f32)) !void {
        var newNeurons = std.ArrayList(Neuron).init(self.allocator);
        var newOutput = std.ArrayList(Value).init(self.allocator);
        for (self.neurons.items) |_neuron| {
            var newNeuron = _neuron;
            try newNeuron.activateInput(input);
            try newNeurons.append(newNeuron);
            try newOutput.append(newNeuron.activation);
        }
        self.neurons.deinit();
        self.output.deinit();
        self.neurons = newNeurons;
        self.output = newOutput;
    }

    pub fn createDeepLayer(
        inputLen: usize,
        layerSize: usize,
        allocator: Allocator,
    ) Layer {
        var neurons = std.ArrayList(Neuron).init(allocator);
        var output = std.ArrayList(Value).init(allocator);
        for (0..layerSize) |_| {
            const newNeuron = Neuron.create(inputLen, allocator);
            output.append(newNeuron.activation) catch {};
            neurons.append(newNeuron) catch {};
        }
        const newLayer = Layer{
            .neurons = neurons,
            .output = output,
            .allocator = allocator,
        };
        return newLayer;
    }

    pub fn activateDeepLayer(self: *Layer, input: std.ArrayList(Value)) !void {
        var newNeurons = std.ArrayList(Neuron).init(self.allocator);
        var newOutput = std.ArrayList(Value).init(self.allocator);
        for (self.neurons.items) |_neuron| {
            var newNeuron = _neuron;
            try newNeuron.activateDeep(input);
            try newNeurons.append(newNeuron);
            try newOutput.append(newNeuron.activation);
        }
        self.neurons.deinit();
        self.output.deinit();
        self.neurons = newNeurons;
        self.output = newOutput;
    }
};
