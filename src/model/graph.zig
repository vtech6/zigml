const _value = @import("value.zig");
const Value = _value.Value;
const std = @import("std");
const utils = @import("utils.zig");
const rl = @import("raylib");

pub fn drawNode(
    value: Value,
    depth: i32,
    nodeHeight: i32,
    nodeWidth: i32,
    nodeX: i32,
    nodeY: i32,
    fontSize: i32,
    margin: i32,
) !void {
    var buf1: [56]u8 = undefined;
    const info1 = try std.fmt.bufPrintZ(
        &buf1,
        "id: {d}\nvalue: {d}\nop: {s}\ngrad: {d}",
        .{
            value.id,
            value.value,
            @tagName(value.op),
            value.gradient,
        },
    );
    const color = utils.randomizeColor(depth);
    rl.drawRectangle(nodeX, nodeY, nodeWidth, nodeHeight, color);

    const textOffset = 5;

    rl.drawText(
        info1,
        nodeX + textOffset,
        nodeY + textOffset,
        fontSize,
        rl.Color.black,
    );

    drawChildren(value, nodeHeight, nodeWidth, margin, depth + 1, fontSize, nodeX, nodeY);
}

fn drawChildren(
    value: Value,
    nodeHeight: i32,
    nodeWidth: i32,
    margin: i32,
    depth: i32,
    fontSize: i32,
    parentX: i32,
    parentY: i32,
) void {
    const childBlockHeight = @divFloor(windowHeight, depth);
    if (value.children.items.len > 0 and depth <= depthLimit) {
        for (value.children.items, 0..) |child, index| {
            const indexInt: i32 = @intCast(index);
            const childX: i32 = ((depth - 1) * (nodeWidth + margin));
            const childY: i32 = parentY - (childBlockHeight * (indexInt)) + 150 - @divFloor(nodeHeight, 2);
            std.debug.print("{d}, {d}, {d}, {d}\n", .{ childY, indexInt, depth, childBlockHeight });
            const childValue = _value.valueMap.get(child);
            const offset = @divFloor(nodeHeight, 2);

            rl.drawLine(
                childX + offset,
                childY + offset,
                parentX + offset,
                parentY + offset,
                rl.Color.black,
            );

            drawNode(
                childValue.?,
                depth,
                nodeHeight,
                nodeWidth,
                childX,
                childY,
                fontSize,
                margin,
            ) catch {};
        }
    }
}

pub const windowWidth: i32 = 560;
pub const windowHeight: i32 = 800;
pub const depthLimit: i32 = 5;
