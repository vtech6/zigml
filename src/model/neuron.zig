const std = @import("std");
const value = @import("value.zig");
const Value = value.Value;
const Allocator = std.mem.Allocator;
pub var neuronMap = std.AutoArrayHashMap(usize, Neuron).init(std.heap.page_allocator);
var randomGenerator = std.rand.DefaultPrng.init(42);
pub var idTracker: usize = 0;

pub const Neuron = struct {
    id: usize,
    bias: usize,
    weights: std.ArrayList(usize),
    output: std.ArrayList(usize),
    activation: Value,
    allocator: Allocator,

    pub fn deinit(self: *const Neuron) !void {
        self.weights.deinit();
        self.output.deinit();
        for (self.activation.children.items) |activationChild| {
            value.valueMap.getPtr(activationChild.id).?.deinit() catch {};
        }
        value.valueMap.getPtr(self.activation.id).?.deinit() catch {};
        self.activation.children.deinit();
    }
    pub fn updateSingleNeuron(self: Value) !void {
        neuronMap.put(self.id, self) catch {};
    }

    pub fn clone(neuron: Neuron) Neuron {
        return Neuron{ .id = neuron.id, .weights = neuron.weights, .bias = neuron.bias, .output = neuron.output, .activation = neuron.activation, .allocator = neuron.allocator };
    }

    pub fn create(inputShape: usize, allocator: std.mem.Allocator) Neuron {
        var weights = std.ArrayList(Value).init(allocator);
        const output = std.ArrayList(Value).init(allocator);

        const rng = randomGenerator.random();

        for (0..inputShape) |_| {
            const randomFloat = rng.float(f32);
            const newValue = Value.create(randomFloat, allocator);
            weights.append(newValue) catch {};
        }
        const randomizedBias = rng.float(f32);
        const newNeuron = Neuron{
            .id = idTracker,
            .weights = weights,
            .bias = Value.create(randomizedBias, allocator),
            .output = output,
            .activation = Value.create(0, allocator),
            .allocator = allocator,
        };

        idTracker += 1;
        neuronMap.put(newNeuron.id, newNeuron) catch {};
        return newNeuron;
    }

    pub fn activateInput(self: *Neuron, input: std.ArrayList(f32)) !void {
        var sumOfOutputs: f32 = 0.0;
        var children = std.ArrayList(Value).init(self.allocator);
        for (input.items, 0..) |element, elementIndex| {
            var newElement = Value.create(element * self.weights.items[elementIndex].value, self.allocator);
            const newOutput = newElement.add(self.bias);
            try children.append(newOutput);
            sumOfOutputs += newOutput.value;
            try self.output.append(newOutput);
            try newElement.deinit();
        }
        var newActivation = Value.create(sumOfOutputs, self.allocator);
        newActivation.rename("Neuron activation");
        newActivation.children = children;
        newActivation.op = value.OPS.activate;
        self.activation = newActivation;
    }

    pub fn activateDeep(self: *Neuron, input: std.ArrayList(Value)) !void {
        var sumOfOutputs: f32 = 0.0;
        var children = std.ArrayList(Value).init(self.allocator);
        for (input.items, 0..) |element, elementIndex| {
            var newElement = element;

            var weighedElement = newElement.multiply(self.weights.items[elementIndex]);
            const newOutput = weighedElement.add(self.bias);
            try children.append(newOutput);
            sumOfOutputs += newOutput.value;
            try self.output.append(newOutput);
            newElement.updateSingleValue() catch {};
            newOutput.updateSingleValue() catch {};
            try weighedElement.deinit();
        }
        var newActivation = Value.create(sumOfOutputs, self.allocator);
        newActivation.rename("Neuron activation");
        newActivation.children = children;
        newActivation.op = value.OPS.activate;
        self.activation = newActivation;
    }
};
