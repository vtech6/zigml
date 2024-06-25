const value = @import("value.zig");
const std = @import("std");
const print = std.debug.print;
test "backpropagation\n" {
    var a = value.Value.create(1);
    var b = value.Value.create(2);
    var c = a.add(&b);
    c.printValue();
    a.printValue();
    try c.backward();
    var readA = value.valueMap.getPtr(a.id);
    readA.?.printValue();
    const count = value.valueMap.count();
    print("Value map count {}\n", .{count});
}
