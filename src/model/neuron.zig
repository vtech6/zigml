const std = @import("std");
const value = @import("value.zig");
const Value = value.Value;

pub const Neuron = struct {
    bias: Value,
    weights: std.ArrayList(Value),
    output: std.ArrayList(Value),

    pub fn freeMemory(self: *Neuron) !void {
        self.weights.deinit();
    }

    pub fn create(allocator: std.mem.Allocator, inputShape: usize) Neuron {
        var weights = std.ArrayList(Value).init(allocator);
        const output = std.ArrayList(Value).init(allocator);

        for (0..inputShape) |_| {
            const newValue = Value.create(10, allocator);
            weights.append(newValue) catch {};
        }
        const newNeuron = Neuron{
            .weights = weights,
            .bias = Value.create(0, allocator),
            .output = output,
        };
        return newNeuron;
    }

    pub fn activateInput(self: *Neuron, input: []f32) !void {
        for (input, 0..) |element, elementIndex| {
            const newOutput = Value.create(element * self.weights[elementIndex].value * self.bias.value);
            self.output.append(newOutput);
        }
    }
};
