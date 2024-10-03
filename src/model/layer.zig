const std = @import("std");
const value = @import("value.zig");
const neuron = @import("neuron.zig");
const Neuron = neuron.Neuron;
const Value = value.Value;
const Allocator = std.mem.Allocator;

pub var layerMap = std.AutoArrayHashMap(usize, Layer).init(
    std.heap.page_allocator,
);
var randomGenerator = std.rand.DefaultPrng.init(42);
pub var idTracker: usize = 0;

pub fn resetState() void {
    idTracker = 0;
}

pub fn cleanup() void {
    const layerMapKeys = layerMap.keys();
    for (layerMapKeys) |layerKey| {
        var _layer = layerMap.get(layerKey).?;
        _layer.deinit() catch {};
    }
    layerMap.clearAndFree();
    resetState();
    neuron.cleanup();
}

pub const Layer = struct {
    output: std.ArrayList(usize),
    neurons: std.ArrayList(usize),
    allocator: Allocator,
    id: usize,

    pub fn deinit(self: *Layer) !void {
        self.output.deinit();
        self.neurons.deinit();
    }

    pub fn createLayer(
        layerSize: usize,
        allocator: Allocator,
    ) Layer {
        var neurons = std.ArrayList(usize).init(allocator);
        var output = std.ArrayList(usize).init(allocator);
        for (0..layerSize) |_| {
            const newNeuron = Neuron.create(layerSize, allocator);
            output.append(newNeuron.activation) catch {};
            neurons.append(newNeuron.id) catch {};
        }

        const newLayer = Layer{
            .neurons = neurons,
            .output = output,
            .allocator = allocator,
            .id = idTracker,
        };

        idTracker += 1;
        newLayer.update();
        return newLayer;
    }

    pub fn update(self: Layer) void {
        layerMap.put(self.id, self) catch {};
    }

    pub fn activateInputLayer(self: *Layer, input: std.ArrayList(f32)) void {
        var activations = std.ArrayList(usize).init(self.allocator);
        for (self.neurons.items) |neuronId| {
            var _neuron = neuron.neuronMap.get(neuronId).?;
            _neuron.activateInput(input) catch {};
            activations.append(_neuron.activation) catch {};
        }
        self.output.deinit();
        self.output = activations;
        self.update();
    }

    pub fn activateDeepLayer(self: *Layer, input: std.ArrayList(usize)) void {
        var activations = std.ArrayList(usize).init(self.allocator);
        for (self.neurons.items) |neuronId| {
            var _neuron = neuron.neuronMap.get(neuronId).?;
            _neuron.activateDeep(input) catch {};
            activations.append(_neuron.activation) catch {};
        }
        if (self.id == 1) {
            std.debug.print("Activation values: {d}, {d}, {d}\n", .{
                activations.items[0],
                activations.items[1],
                activations.items[2],
            });
        }

        std.debug.print("activations length for layer {d}: {d}\n", .{
            self.id,
            activations.items.len,
        });
        self.output.deinit();
        self.output = activations;
        self.update();
    }
};
