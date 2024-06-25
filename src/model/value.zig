const std = @import("std");
const Allocator = std.mem.Allocator;

pub var idTracker: u8 = 0;

pub fn resetState() void {
    idTracker = 0;
}
pub const SomeError = error{};

pub const Value = struct {
    id: u8 = 0,
    value: f32,
    gradient: f32 = 1.0,
    children: std.ArrayList(Value),
    label: []const u8 = "value",
    op: []const u8 = "value",

    //Neural network methods

    pub fn backward(self: *Value) !void {
        self.gradient = 1.0;
    }

    pub fn create(value: f32) Value {
        const newNode = Value{ .id = idTracker, .value = value, .previous = .{ 0, 0 } };
        idTracker += 1;
        return newNode;
    }

    pub fn rename(self: *Value, label: []const u8) void {
        self.label = label;
    }

    pub fn setChildren(self: *Value, children: []Value) void {
        self.children = children;
    }

    //Math methods

    pub fn add(self: *Value, _b: *Value) Value {
        var newValue = Value.create(self.value + _b.value);
        const children = [2]u8{ self.id, _b.id };
        newValue.setChildren(children);
        return newValue;
    }

    pub fn multiply(self: *Value, multiplier: *Value) Value {
        var newValue = Value.create(self.value * multiplier.value);
        newValue.setChildren();
    }

    //Utility methods

    pub fn resetGradient(self: *Value) void {
        self.gradient = 1.0;
    }

    pub fn printChildren(self: *Value) void {
        std.debug.print("Children len: {d}", .{self.previous.len});
    }

    pub fn printValue(self: Value) void {
        std.debug.print("Name: {s}, Value: {d}\n", .{ self.label, self.value });
    }
};
