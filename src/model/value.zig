const std = @import("std");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
pub var valueMap = std.AutoArrayHashMap(usize, Value).init(std.heap.page_allocator);
pub var backpropagationOrder = std.ArrayList(usize).init(std.heap.page_allocator);
pub const OPS = enum { add, init, multiply };

pub var idTracker: usize = 0;

pub fn resetState() void {
    idTracker = 0;
    valueMap.clearAndFree();
    backpropagationOrder.clearAndFree();
}

pub const SomeError = error{};

pub const Value = struct {
    id: usize = 0,
    value: f32,
    gradient: f32 = 1.0,
    children: ArrayList(Value),
    label: []const u8 = "value",
    op: OPS = OPS.init,
    allocator: Allocator,

    pub fn deinit(self: *Value) !void {
        self.children.deinit();
    }

    pub fn prepareBackpropagation(self: Value) void {
        const _self = [1]usize{self.id};
        if (std.mem.containsAtLeast(usize, backpropagationOrder.items, 1, &_self) == false) {
            backpropagationOrder.append(self.id) catch {};
        }
        for (self.children.items) |node| {
            const nodes = [1]usize{node.id};
            if (std.mem.containsAtLeast(usize, backpropagationOrder.items, 1, &nodes) == false) {
                backpropagationOrder.append(node.id) catch {};
                prepareBackpropagation(node);
            }
        }
    }

    pub fn backpropagate() void {
        for (backpropagationOrder.items) |node| {
            var _value = valueMap.getPtr(node).?;
            _value.backward() catch {};
        }
    }

    pub fn backward(self: *Value) !void {
        std.debug.print("Calling Backward {d}, op: {any}\n", .{ self.value, self.op });
        switch (self.op) {
            .add => backwardAdd(self),
            .multiply => backwardMultiply(self),
            .init => return,
        }
    }

    pub fn setGradient(self: *Value, gradient: f32) void {
        self.gradient = gradient;
        self.updateSingleValue() catch {};
    }

    //Neural network methods

    pub fn create(value: f32, allocator: Allocator) Value {
        const newNode = Value{ .id = idTracker, .value = value, .children = std.ArrayList(Value).init(allocator), .allocator = allocator };
        idTracker += 1;
        valueMap.put(newNode.id, newNode) catch {};
        return newNode;
    }

    pub fn rename(self: *Value, label: []const u8) void {
        self.label = label;
    }

    //Math methods

    pub fn add(self: Value, _b: Value) Value {
        var newValue = Value.create(self.value + _b.value, self.allocator);
        newValue.op = OPS.add;
        const children = [2]Value{ self, _b };
        var newChildren = std.ArrayList(Value).init(self.allocator);
        for (children) |node| {
            newChildren.append(node) catch {};
        }
        newValue.children = newChildren;
        newValue.updateSingleValue() catch {};
        return newValue;
    }

    pub fn updateSingleValue(self: Value) !void {
        valueMap.put(self.id, self) catch {};
    }

    fn backwardAdd(self: *Value) void {
        std.debug.print("BACKWARD ADD\n", .{});
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

    fn updateValueMap(self: *Value) void {
        const children = self.children.items;
        const newVal = Value{ .id = self.id, .value = self.value, .gradient = self.gradient, .children = self.children, .label = self.label, .op = self.op, .allocator = self.allocator };
        valueMap.put(self.id, newVal) catch {};
        valueMap.put(children[0].id, children[0]) catch {};
        valueMap.put(children[1].id, children[1]) catch {};
    }

    pub fn multiply(self: Value, multiplier: Value) Value {
        var newValue = Value.create(self.value * multiplier.value, self.allocator);
        newValue.op = OPS.multiply;
        const children = [2]Value{ self, multiplier };
        var newChildren = std.ArrayList(Value).init(self.allocator);
        for (children) |node| {
            newChildren.append(node) catch {};
        }
        newValue.children = newChildren;
        newValue.updateSingleValue() catch {};

        return newValue;
    }

    pub fn backwardMultiply(self: *Value) void {
        std.debug.print("BACKWARD MULTIPLY\n", .{});
        var newChildren = std.ArrayList(Value).init(self.allocator);
        var newA = valueMap.get(self.children.items[0].id);
        var newB = valueMap.get(self.children.items[1].id);
        newA.?.setGradient(self.gradient * newB.?.value);
        newB.?.setGradient(self.gradient * newA.?.value);
        newChildren.append(newA.?) catch {};
        newChildren.append(newB.?) catch {};
        self.children.deinit();
        self.children = newChildren;
        self.updateValueMap();
    }

    //Utility methods

    pub fn resetGradient(self: *Value) void {
        self.gradient = 1.0;
    }

    pub fn printChildren(self: *Value) void {
        std.debug.print("Children len: {d}", .{self.children.len});
    }

    pub fn printValue(self: Value) void {
        std.debug.print("Name: {s}, Value: {d}\n", .{ self.label, self.value });
    }
};
