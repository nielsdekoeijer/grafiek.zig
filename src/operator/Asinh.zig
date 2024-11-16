const utils = @import("../core/utils.zig");
const std = @import("std");

pub fn Asinh(comptime Self: type, comptime inp: anytype, comptime out: anytype) type {
    comptime {
        return struct {
            pub fn process(self: *Self) void {
                inline for (0..@TypeOf(@field(self.inner.tensors, out))._len()) |i| {
                    @field(self.inner.tensors, out).data[i] = std.math.asinh(@field(self.inner.tensors, inp).data[i]);
                }
            }
        };
    }
}

const OperatorType = @import("../operator/OperatorType.zig").OperatorType;
const TensorType = @import("../tensor/TensorType.zig").TensorType;
const Make = @import("../core/Make.zig").Make;

test "operator.Asinh" {
    var model = Make(
        .{
            .tensors = .{
                .{
                    .name = "Input",
                    .type = f32,
                    .shape = .{ 2, 3 },
                    .tensor = TensorType.Input,
                    .params = .{
                        .data = .{ 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 },
                    },
                },
                .{
                    .name = "Output",
                    .type = f32,
                    .shape = .{ 2, 3 },
                    .tensor = TensorType.Output,
                    .params = {},
                },
            },
            .operators = .{
                .{
                    .name = "Asinh",
                    .op = OperatorType.Asinh,
                    .type = f32,
                    .params = .{
                        .inp_0 = "Input",
                        .inp_0_shape = .{ 2, 3 },
                        .out_0 = "Output",
                        .out_0_shape = .{ 2, 3 },
                    },
                },
            },
        },
    ).make();

    model.process();

    try std.testing.expect(model.inner.tensors.Output.data[0] == std.math.asinh(@as(f32, 2.0)));
    try std.testing.expect(model.inner.tensors.Output.data[1] == std.math.asinh(@as(f32, 3.0)));
    try std.testing.expect(model.inner.tensors.Output.data[2] == std.math.asinh(@as(f32, 4.0)));
    try std.testing.expect(model.inner.tensors.Output.data[3] == std.math.asinh(@as(f32, 5.0)));
    try std.testing.expect(model.inner.tensors.Output.data[4] == std.math.asinh(@as(f32, 6.0)));
    try std.testing.expect(model.inner.tensors.Output.data[5] == std.math.asinh(@as(f32, 7.0)));
}
