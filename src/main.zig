const std = @import("std");
const Make = @import("core/Make.zig").Make;

// MatMul
// Split
// Batch Normalization
// Squeeze
// Slice
// LSTM
// InstanceNormalization
// Concat
// Gather

pub fn main() !void {
    var model = Make(@import("models/simple.zig").model).make();

    model.inner.tensors.Input.data[0] = 1.0;
    model.inner.tensors.Input.data[1] = 2.0;
    model.inner.tensors.Input.data[2] = 3.0;
    model.inner.tensors.Input.data[3] = 4.0;
    model.inner.tensors.Input.data[4] = 5.0;
    model.inner.tensors.Input.data[5] = 6.0;

    model.process();

    std.debug.print("{}\n", .{model.inner.tensors.Input.get(.{ 0, 0 })});
    std.debug.print("{}\n", .{model.inner.tensors.Input.get(.{ 0, 1 })});
    std.debug.print("{}\n", .{model.inner.tensors.Input.get(.{ 0, 2 })});
    std.debug.print("{}\n", .{model.inner.tensors.Input.get(.{ 1, 0 })});
    std.debug.print("{}\n", .{model.inner.tensors.Input.get(.{ 1, 1 })});
    std.debug.print("{}\n", .{model.inner.tensors.Input.get(.{ 1, 2 })});

    std.debug.print("{}\n", .{model.inner.tensors.Output.get(.{ 0, 0 })});
    std.debug.print("{}\n", .{model.inner.tensors.Output.get(.{ 0, 1 })});
    std.debug.print("{}\n", .{model.inner.tensors.Output.get(.{ 1, 0 })});
    std.debug.print("{}\n", .{model.inner.tensors.Output.get(.{ 1, 1 })});
    std.debug.print("{}\n", .{model.inner.tensors.Output.get(.{ 2, 0 })});
    std.debug.print("{}\n", .{model.inner.tensors.Output.get(.{ 2, 1 })});
}
