const std = @import("std");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const rl = @import("raylib");
const utils = @import("utils.zig");
const graph = @import("graph.zig");
const math = std.math;

pub var valueMap = std.AutoArrayHashMap(usize, Value).init(std.heap.page_allocator);
pub var backpropagationOrder = std.ArrayList(usize).init(std.heap.page_allocator);
pub const OPS = enum {
    add,
    init,
    multiply,
    activate,
    tanh,
    pow,
};

pub var idTracker: usize = 0;

pub fn resetState() void {
    idTracker = 0;
}

pub fn cleanup() void {
    const valueMapKeys = valueMap.keys();
    for (valueMapKeys) |valueKey| {
        var _value = valueMap.get(valueKey).?;
        _value.deinit();
    }
    valueMap.clearAndFree();
    resetState();
}

pub const Value = struct {
    id: usize = 0,
    value: f32,
    gradient: f32 = 0.0,
    children: ArrayList(usize),
    label: []const u8 = "value",
    op: OPS = OPS.init,
    allocator: Allocator,

    pub fn deinit(self: *Value) void {
        self.children.deinit();
    }

    pub fn create(value: f32, allocator: Allocator) Value {
        const newChildren = std.ArrayList(usize).init(allocator);
        const newNode = Value{
            .id = idTracker,
            .value = value,
            .children = newChildren,
            .allocator = allocator,
        };
        idTracker += 1;
        newNode.update();
        return newNode;
    }

    pub fn backpropagate(self: Value) void {
        self.calculateGradient();
        if (self.children.items.len > 0) {
            for (self.children.items) |child| {
                const childValue = valueMap.get(child).?;
                childValue.backpropagate();
            }
        }
    }

    pub fn calculateGradient(self: Value) void {
        switch (self.op) {
            OPS.add => self.backwardAdd(),
            OPS.multiply => self.backwardMultiply(),
            OPS.init => {},
            OPS.activate => {},
            OPS.tanh => self.backwardTanh(),
            OPS.pow => self.backwardPow(),
        }
    }

    //Neural network methods

    pub fn rename(self: *Value, label: []const u8) void {
        self.label = label;
    }

    //Math methods

    pub fn add(self: Value, _b: Value) Value {
        var newValue = Value.create(self.value + _b.value, self.allocator);
        newValue.children.append(self.id) catch {};
        newValue.children.append(_b.id) catch {};
        newValue.op = OPS.add;
        newValue.update();
        return newValue;
    }

    pub fn multiply(self: Value, _b: Value) Value {
        var newValue = Value.create(self.value * _b.value, self.allocator);
        newValue.children.append(self.id) catch {};
        newValue.children.append(_b.id) catch {};
        newValue.op = OPS.multiply;
        newValue.update();
        return newValue;
    }

    pub fn tanh(self: Value) Value {
        const x = self.value;
        const result = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1);
        var resultValue = Value.create(result, self.allocator);
        resultValue.op = OPS.tanh;
        resultValue.children.append(self.id) catch {};
        resultValue.update();
        return resultValue;
    }

    pub fn backwardTanh(self: Value) void {
        var newValue = valueMap.get(self.children.items[0]).?;
        newValue.gradient += (1 - math.pow(f32, self.value, 2)) * self.gradient;
        newValue.update();
    }

    fn backwardAdd(self: Value) void {
        for (self.children.items) |child| {
            var childValue = valueMap.get(child).?;
            childValue.gradient += 1.0 * self.gradient;
            childValue.update();
        }
    }

    fn backwardMultiply(self: Value) void {
        const children = self.children.items;
        var child1 = valueMap.get(children[0]).?;
        var child2 = valueMap.get(children[1]).?;

        child1.gradient += child2.value * self.gradient;
        child2.gradient += child1.value * self.gradient;

        child1.update();
        child2.update();
    }
    pub fn pow(self: Value, power: f32) Value {
        const _power = Value.create(power, self.allocator);
        var output = Value.create(math.pow(f32, self.value, power), self.allocator);
        output.children.append(self.id) catch {};
        output.children.append(_power.id) catch {};
        output.op = OPS.pow;
        output.update();
        return output;
    }

    //Utility methods
    fn backwardPow(self: Value) void {
        var childValue = valueMap.get(self.children.items[0]).?;
        const powerValue = valueMap.get(self.children.items[1]).?;
        childValue.gradient += powerValue.value * (math.pow(f32, childValue.value, powerValue.value - 1)) * self.gradient;
        childValue.update();
    }

    pub fn resetGradient(self: *Value) void {
        self.gradient = 0.0;
        self.update();
    }

    pub fn printValue(self: Value) void {
        std.debug.print("Name: {s}, Value: {d}\n", .{ self.label, self.value });
    }

    pub fn visualize(self: Value) void {
        rl.initWindow(graph.windowWidth, graph.windowHeight, "raylib-zig [core] example - basic window");
        defer rl.closeWindow(); // Close window and OpenGL context

        rl.setTargetFPS(1); // Set our game to run at 60 frames-per-second
        while (!rl.windowShouldClose()) { // Detect window close button or ESC key
            rl.beginDrawing();
            defer rl.endDrawing();
            rl.clearBackground(rl.Color.white);
            const nodeWidth = 64;
            const nodeHeight: i32 = 64;
            const margin: i32 = 8;
            const nodeX = margin;
            const nodeY = (graph.windowHeight / 2) - (nodeHeight / 2);
            const depth = 1;
            const fontSize = 10;

            graph.drawNode(self, depth, nodeHeight, nodeWidth, nodeX, nodeY, fontSize, margin) catch {};
        }
    }

    pub fn update(self: Value) void {
        valueMap.put(self.id, self) catch {};
    }
};
