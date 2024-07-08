const std = @import("std");
const value = @import("value.zig");
const Value = value.Value;
const Allocator = std.mem.Allocator;
pub var neuronMap = std.AutoArrayHashMap(usize, Neuron).init(std.heap.page_allocator);
var randomGenerator = std.rand.DefaultPrng.init(42);
pub var idTracker: usize = 0;

pub const Neuron = struct {
    id: usize,
    bias: Value,
    weights: std.ArrayList(Value),
    output: std.ArrayList(Value),
    activation: Value,
    allocator: Allocator,

    pub fn deinit(self: *const Neuron) !void {
        self.weights.deinit();
        self.output.deinit();
        self.activation.children.deinit();
    }
    pub fn updateSingleNeuron(self: Value) !void {
        neuronMap.put(self.id, self) catch {};
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
            const newOutput = Value.create(element * self.weights.items[elementIndex].value + self.bias.value, self.allocator);
            try children.append(newOutput);
            sumOfOutputs += newOutput.value;
            try self.output.append(newOutput);
        }
        var newActivation = Value.create(sumOfOutputs, self.allocator);
        newActivation.rename("Neuron activation");
        newActivation.children = children;
        self.activation = newActivation;
    }
};
