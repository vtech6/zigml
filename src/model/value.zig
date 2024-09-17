const std = @import("std");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const rl = @import("raylib");

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
        rl.initWindow(windowWidth, windowHeight, "raylib-zig [core] example - basic window");
        defer rl.closeWindow(); // Close window and OpenGL context

        rl.setTargetFPS(1); // Set our game to run at 60 frames-per-second
        while (!rl.windowShouldClose()) { // Detect window close button or ESC key
            // Update
            //----------------------------------------------------------------------------------
            // TODO: Update your variables here
            //----------------------------------------------------------------------------------

            // Draw
            //----------------------------------------------------------------------------------
            rl.beginDrawing();
            defer rl.endDrawing();
            rl.clearBackground(rl.Color.white);
            const nodeWidth = 50;
            const nodeHeight: i32 = 50;
            const margin: i32 = 10;
            const nodeX = margin;
            const nodeY = (windowHeight / 2) - (nodeHeight / 2);
            const depth = 1;
            const fontSize = 10;

            self.drawChildren(
                depth,
                nodeHeight,
                nodeWidth,
                nodeX,
                nodeY,
                fontSize,
                margin,
            ) catch {};
            //----------------------------------------------------------------------------------
        }
    }

    pub fn drawChildren(
        self: Value,
        depth: i32,
        nodeHeight: i32,
        nodeWidth: i32,
        nodeX: i32,
        nodeY: i32,
        fontSize: i32,
        margin: i32,
    ) !void {
        var buf: [20]u8 = undefined;
        const terminatedValue = try std.fmt.bufPrintZ(&buf, "value: {d}\nid: {d}", .{ self.value, self.id });
        const color = randomizeColor(depth);
        rl.drawRectangle(
            nodeX,
            nodeY,
            nodeWidth,
            nodeHeight,
            color,
        );

        const textOffset = 5;

        rl.drawText(
            terminatedValue,
            nodeX + textOffset,
            nodeY + textOffset,
            fontSize,
            rl.Color.black,
        );

        if (self.children.items.len > 0) {
            for (self.children.items, 0..) |child, index| {
                const indexInt: i32 = @intCast(index);
                const childY: i32 = (indexInt * (nodeHeight + margin)) + nodeY;
                const childX: i32 = (depth * (nodeWidth + margin)) + margin;
                const childValue = valueMap.get(child);
                try childValue.?.drawChildren(
                    depth + 1,
                    nodeHeight,
                    nodeWidth,
                    childX,
                    childY,
                    fontSize,
                    margin,
                );
            }
        }
    }
};

fn randomizeColor(depth: i32) rl.Color {
    const castDepth: u32 = @intCast(depth);

    const red: u32 = (castDepth * 60) % 255;
    const green: u32 = (castDepth * 80) % 255;
    const blue: u32 = (castDepth * 100) % 255;

    const redSqueezed: u8 = @intCast(red);
    const greenSqueezed: u8 = @intCast(green);
    const blueSqueezed: u8 = @intCast(blue);

    const color = rl.Color{
        .r = redSqueezed,
        .g = greenSqueezed,
        .b = blueSqueezed,
        .a = 255,
    };
    return color;
}

const windowWidth: i32 = 800;
const windowHeight: i32 = 450;
