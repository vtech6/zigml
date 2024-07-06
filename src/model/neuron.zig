const std = @import("std");
const value = @import("value.zig");
const Value = value.Value;
const Allocator = std.mem.Allocator;

pub const Neuron = struct {
    bias: Value,
    weights: std.ArrayList(Value),
    output: std.ArrayList(Value),
    allocator: Allocator,

    pub fn freeMemory(self: *Neuron) !void {
        self.weights.deinit();
        self.output.deinit();
    }

    pub fn create(inputShape: usize, allocator: std.mem.Allocator) Neuron {
        var weights = std.ArrayList(Value).init(allocator);
        const output = std.ArrayList(Value).init(allocator);

        for (0..inputShape) |_| {
            const randomFloat = std.crypto.random.float(f32);
            const newValue = Value.create(randomFloat, allocator);
            weights.append(newValue) catch {};
        }
        const newNeuron = Neuron{
            .weights = weights,
            .bias = Value.create(1, allocator),
            .output = output,
            .allocator = allocator,
        };
        return newNeuron;
    }

    pub fn activateInput(self: *Neuron, input: [2]f32) !void {
        for (input, 0..) |element, elementIndex| {
            const newOutput = Value.create(element * self.weights.items[elementIndex].value * self.bias.value, self.allocator);
            try self.output.append(newOutput);
        }
    }
};
