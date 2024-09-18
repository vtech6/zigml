const std = @import("std");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const rl = @import("raylib");
const utils = @import("utils.zig");
const graph = @import("graph.zig");
const math = std.math;

pub var valueMap = std.AutoArrayHashMap(usize, Value).init(std.heap.page_allocator);
pub var backpropagationOrder = std.ArrayList(usize).init(std.heap.page_allocator);
pub const OPS = enum { add, init, multiply, activate, tanh };

pub var idTracker: usize = 0;

pub fn resetState() void {
    idTracker = 0;
}

pub const Value = struct {
    id: usize = 0,
    value: f32,
    gradient: f32 = 1.0,
    children: ArrayList(usize),
    label: []const u8 = "value",
    op: OPS = OPS.init,
    allocator: Allocator,

    pub fn deinit(self: *Value) !void {
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
        valueMap.put(newNode.id, newNode) catch {};
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
            OPS.tanh => {},
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
        valueMap.put(newValue.id, newValue) catch {};
        return newValue;
    }

    pub fn multiply(self: Value, _b: Value) Value {
        var newValue = Value.create(self.value * _b.value, self.allocator);
        newValue.children.append(self.id) catch {};
        newValue.children.append(_b.id) catch {};
        newValue.op = OPS.multiply;
        valueMap.put(newValue.id, newValue) catch {};
        return newValue;
    }

    pub fn tanh(self: Value) Value {
        const x = self.value;
        const result = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1);
        var resultValue = Value.create(result, self.allocator);
        resultValue.op = OPS.tanh;
        resultValue.children.append(self.id) catch {};
        valueMap.put(resultValue.id, resultValue) catch {};
        std.debug.print("{d}", .{resultValue.value});
        return resultValue;
    }

    pub fn backwardTanh(self: Value) void {
        var newValue = self;
        newValue.gradient += (1 - math.pow(f32, self.value, 2)) * self.gradient;
        valueMap.put(self.id, newValue);
    }

    pub fn updateSingleValue(self: Value) !void {
        valueMap.put(self.id, self) catch {};
    }

    fn backwardAdd(self: Value) void {
        for (self.children.items) |child| {
            var childValue = valueMap.get(child).?;
            childValue.gradient += self.gradient;
            valueMap.put(childValue.id, childValue) catch {};
        }
    }

    fn backwardMultiply(self: Value) void {
        const children = self.children.items;
        var child1 = valueMap.get(children[0]).?;
        var child2 = valueMap.get(children[1]).?;

        child1.gradient += child2.value * self.gradient;
        child2.gradient += child1.value * self.gradient;

        valueMap.put(child1.id, child1) catch {};
        valueMap.put(child2.id, child2) catch {};
    }

    //Utility methods

    pub fn resetGradient(self: *Value) void {
        self.gradient = 1.0;
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
};
