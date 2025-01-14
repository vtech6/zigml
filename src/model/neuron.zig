const std = @import("std");
const generateRandomFloat = @import("utils.zig").generateRandomFloat;
const ArrayList = std.ArrayList;
const value = @import("value.zig");
const Value = value.Value;
const Allocator = std.mem.Allocator;
const Activation = @import("activation.zig").Activation;
pub var neuronMap = std.AutoArrayHashMap(usize, Neuron).init(std.heap.page_allocator);
pub var idTracker: usize = 0;

pub fn resetState() void {
    idTracker = 0;
}

pub fn cleanup() void {
    const neuronMapKeys = neuronMap.keys();
    for (neuronMapKeys) |neuronKey| {
        var _neuron = neuronMap.get(neuronKey).?;
        _neuron.deinit() catch {};
    }
    neuronMap.clearAndFree();
    resetState();
    value.cleanup();
}

pub const Neuron = struct {
    id: usize,
    bias: usize,
    weights: ArrayList(usize),
    output: ArrayList(usize),
    activation: usize,
    activationFunction: Activation = Activation.tanh,
    allocator: Allocator,

    pub fn deinit(self: *Neuron) !void {
        self.weights.deinit();
        self.output.deinit();
    }

    pub fn create(inputShape: usize, allocator: std.mem.Allocator) Neuron {
        var weights = std.ArrayList(usize).init(allocator);
        const output = std.ArrayList(usize).init(allocator);

        for (0..inputShape) |_| {
            const randomFloat = generateRandomFloat();
            const newValue = Value.create(randomFloat, allocator);
            weights.append(newValue.id) catch {};
        }
        const randomizedBias = generateRandomFloat();
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
        newNeuron.update();
        return newNeuron;
    }

    pub fn update(self: Neuron) void {
        neuronMap.put(self.id, self) catch {};
    }

    pub fn clone(neuron: Neuron) Neuron {
        return Neuron{ .id = neuron.id, .weights = neuron.weights, .bias = neuron.bias, .output = neuron.output, .activation = neuron.activation, .allocator = neuron.allocator };
    }

    pub fn activateInput(self: *Neuron, input: std.ArrayList(f32)) !void {
        var sumOfOutputs: f32 = 0.0;
        var children = std.ArrayList(usize).init(self.allocator);
        for (input.items, 0..) |element, elementIndex| {
            const weightValue = value.valueMap.get(self.weights.items[elementIndex]).?;
            const newElement = Value.create(
                element * weightValue.value,
                self.allocator,
            );
            try children.append(newElement.id);
            sumOfOutputs += newElement.value;
            try self.output.append(newElement.id);
        }
        const bias = value.valueMap.get(self.bias).?;
        var sumOfOutputsValue = Value.create(sumOfOutputs, self.allocator);
        sumOfOutputsValue.op = value.OPS.add;
        sumOfOutputsValue.children.deinit();
        sumOfOutputsValue.children = children;
        sumOfOutputsValue.update();
        const sumOfOutputsValueWithBias = sumOfOutputsValue.add(bias);
        var newActivation: Value = undefined;
        switch (self.activationFunction) {
            Activation.tanh => newActivation = sumOfOutputsValueWithBias.tanh(),
            Activation.exp => {},
        }
        newActivation.rename("Neuron activation");
        newActivation.update();
        self.activation = newActivation.id;
        self.update();
    }

    pub fn activateDeep(self: *Neuron, input: std.ArrayList(usize)) !void {
        var sumOfOutputs: f32 = 0.0;
        var children = std.ArrayList(usize).init(self.allocator);
        for (self.weights.items, 0..) |weight, weightIndex| {
            const elementValue = value.valueMap.get(input.items[weightIndex]).?;
            const weightValue = value.valueMap.get(weight).?;
            const newElement = elementValue.multiply(weightValue);
            try children.append(newElement.id);
            sumOfOutputs += newElement.value;
            try self.output.append(newElement.id);
        }
        const bias = value.valueMap.get(self.bias).?;
        var sumOfOutputsValue = Value.create(sumOfOutputs, self.allocator);
        sumOfOutputsValue.op = value.OPS.add;
        sumOfOutputsValue.children.deinit();
        sumOfOutputsValue.children = children;
        sumOfOutputsValue.update();
        const sumOfOutputsValueWithBias = sumOfOutputsValue.add(bias);
        var newActivation: Value = undefined;
        switch (self.activationFunction) {
            Activation.tanh => newActivation = sumOfOutputsValueWithBias.tanh(),
            Activation.exp => {},
        }

        newActivation.rename("Neuron activation");
        newActivation.update();
        self.activation = newActivation.id;
        self.update();
    }
    pub fn print(self: Neuron) void {
        const bias = value.valueMap.get(self.bias).?;
        const activation = value.valueMap.get(self.activation).?;
        std.debug.print("NeuronId: {d}, fwid: {d}, bId: {d}, bv: {d}, bg: {d}, a: {d}\n", .{
            self.id,
            self.weights.items[0],
            bias.id,
            bias.value,
            bias.gradient,
            activation.value,
        });
    }
};
