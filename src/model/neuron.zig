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
    activation: usize,
    allocator: Allocator,

    pub fn deinit(self: Neuron) !void {
        self.weights.deinit();
        self.output.deinit();
    }

    pub fn update(self: Value) !void {
        neuronMap.put(self.id, self) catch {};
    }

    pub fn clone(neuron: Neuron) Neuron {
        return Neuron{ .id = neuron.id, .weights = neuron.weights, .bias = neuron.bias, .output = neuron.output, .activation = neuron.activation, .allocator = neuron.allocator };
    }

    pub fn create(inputShape: usize, allocator: std.mem.Allocator) Neuron {
        var weights = std.ArrayList(usize).init(allocator);
        const output = std.ArrayList(usize).init(allocator);

        const rng = randomGenerator.random();

        for (0..inputShape) |_| {
            const randomFloat = rng.float(f32);
            const newValue = Value.create(randomFloat, allocator);
            weights.append(newValue.id) catch {};
        }
        const randomizedBias = rng.float(f32);
        const biasValue = Value.create(randomizedBias, allocator);
        const activationValue = Value.create(0, allocator);
        const newNeuron = Neuron{
            .id = idTracker,
            .weights = weights,
            .bias = biasValue.id,
            .output = output,
            .activation = activationValue.id,
            .allocator = allocator,
        };

        idTracker += 1;
        neuronMap.put(newNeuron.id, newNeuron) catch {};
        return newNeuron;
    }

    pub fn activateInput(self: *Neuron, input: std.ArrayList(f32)) !void {
        var sumOfOutputs: f32 = 0.0;
        var children = std.ArrayList(usize).init(self.allocator);
        for (input.items, 0..) |element, elementIndex| {
            const weightValue = value.valueMap.get(self.weights.items[elementIndex]).?;
            const bias = value.valueMap.get(self.bias).?;
            var newElement = Value.create(
                element * weightValue.value,
                self.allocator,
            );
            var newOutput = newElement.add(bias);
            try children.append(newOutput.id);
            sumOfOutputs += newOutput.value;
            try self.output.append(newOutput.id);
            try newElement.deinit();
            try newOutput.deinit();
        }
        var newActivation = Value.create(sumOfOutputs, self.allocator);
        newActivation.rename("Neuron activation");
        newActivation.children = children;
        newActivation.op = value.OPS.activate;
        newActivation.update();
        self.activation = newActivation.id;
        try newActivation.deinit();
    }

    pub fn activateDeep(self: *Neuron, input: std.ArrayList(Value)) !void {
        var sumOfOutputs: f32 = 0.0;
        var children = std.ArrayList(usize).init(self.allocator);
        for (input.items, 0..) |element, elementIndex| {
            var newElement = element;
            const weight = value.valueMap.get(self.weights.items[elementIndex]).?;
            const bias = value.valueMap.get(self.bias).?;
            var weighedElement = newElement.multiply(weight);
            const newOutput = weighedElement.add(bias);
            try children.append(newOutput);
            sumOfOutputs += newOutput.value;
            try self.output.append(newOutput.id);
            newElement.update() catch {};
            newOutput.update() catch {};
        }
        var newActivation = Value.create(sumOfOutputs, self.allocator);
        newActivation.rename("Neuron activation");
        newActivation.children = children;
        newActivation.op = value.OPS.activate;
        newActivation.update();
        self.activation = newActivation.id;
    }
};
