const rl = @import("raylib");
pub fn randomizeColor(depth: i32) rl.Color {
    const castDepth: u32 = @intCast(depth);

    const red: u32 = (castDepth * 160) % 255;
    const green: u32 = (castDepth * 120) % 255;
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
