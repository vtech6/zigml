const std = @import("std");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const rl = @import("raylib");
const utils = @import("utils.zig");
const graph = @import("graph.zig");

pub var valueMap = std.AutoArrayHashMap(usize, Value).init(std.heap.page_allocator);
pub var backpropagationOrder = std.ArrayList(usize).init(std.heap.page_allocator);
pub const OPS = enum { add, init, multiply, activate };

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
        const newNode = Value{ .id = idTracker, .value = value, .children = newChildren, .allocator = allocator };
        idTracker += 1;
        valueMap.put(newNode.id, newNode) catch {};
        return newNode;
    }

    pub fn backpropagate() void {}

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

    pub fn updateSingleValue(self: Value) !void {
        valueMap.put(self.id, self) catch {};
    }

    fn backwardAdd(self: *Value) void {
        var newChildren = std.ArrayList(Value).init(self.allocator);
        var newA = valueMap.get(self.children.items[0].id).?;
        var newB = valueMap.get(self.children.items[1].id).?;
        newA.setGradient(self.gradient);
        newB.setGradient(self.gradient);
        newChildren.append(newA) catch {};
        newChildren.append(newB) catch {};
        self.children.deinit();
        self.children = newChildren;
        self.updateValueMap();
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
            const nodeWidth = 56;
            const nodeHeight: i32 = 56;
            const margin: i32 = 10;
            const nodeX = margin;
            const nodeY = (graph.windowHeight / 2) - (nodeHeight / 2);
            const depth = 1;
            const fontSize = 10;

            graph.drawNode(self, depth, nodeHeight, nodeWidth, nodeX, nodeY, fontSize, margin) catch {};
        }
    }
};
