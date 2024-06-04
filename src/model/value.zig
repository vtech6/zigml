const std = @import("std");

var idTracker: i8 = 0;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();
var visited = std.AutoArrayHashMap(i8, *const [3]i8).init(allocator);
pub var topo = std.ArrayList(*Value).init(allocator);

pub const SomeError = error{};

pub const Value = struct {
    id: i8 = 0,
    value: f32,
    gradient: f32 = 1.0,
    previous: []const *Value = &[_]*Value{},
    label: []const u8 = "value",
    op: []const u8 = "value",

    pub fn backward(self: *Value) !void {
        buildTopo(self) catch {};
        self.gradient = 1.0;
    }

    fn buildTopo(value: *Value) !void {
        if (visited.get(value.id) == null) {
            const ids = getPreviousIds();
            visited.put(value.id, ids) catch {};
            for (value.previous) |childValue| {
                try buildTopo(childValue);
            }
            topo.append(value) catch {};
        }
    }

    pub fn create(value: f32) Value {
        idTracker += 1;
        return Value{ .id = idTracker, .value = value };
    }

    pub fn getPreviousIds() *const [3]i8 {
        const ids = [_]i8{ 2, 3, 4 };
        return &ids;
    }

    pub fn rename(self: *Value, label: []const u8) void {
        self.label = label;
    }

    pub fn setChildren(self: *Value, children: []const *Value) void {
        self.previous = children;
    }

    pub fn add(self: *Value, _b: *Value) Value {
        var newValue = Value.create(self.value + _b.value);
        const children = [2]*Value{ self, _b };
        newValue.setChildren(&children);
        return newValue;
    }

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

pub fn main() !void {}
