const utils = @import("../core/utils.zig");
const std = @import("std");

pub fn Transpose(comptime Self: type, comptime inp: anytype, comptime out: anytype, comptime perm: anytype) type {
    comptime {
        return struct {
            pub fn process(self: *Self) void {
                inline for (0..@TypeOf(@field(self.inner.tensors, out))._len()) |i| {
                    const j = utils.comptimeIndexAsShape(@TypeOf(@field(self.inner.tensors, out))._shape(), i);
                    @field(self.inner.tensors, out).set(j, @field(self.inner.tensors, inp).get(utils.comptimeIndexPermute(j, perm)));
                }
            }
        };
    }
}

const OperatorType = @import("../operator/OperatorType.zig").OperatorType;
const TensorType = @import("../tensor/TensorType.zig").TensorType;
const Make = @import("../core/Make.zig").Make;

test "operator.Transpose" {
    var model = Make(
        .{
            .tensors = .{
                .{
                    .name = "Input",
                    .type = f32,
                    .shape = .{ 2, 3 },
                    .tensor = TensorType.Input,
                    .params = .{
                        .data = .{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 },
                    },
                },
                .{
                    .name = "Output",
                    .type = f32,
                    .shape = .{ 3, 2 },
                    .tensor = TensorType.Output,
                    .params = {},
                },
            },
            .operators = .{
                .{
                    .name = "Transpose",
                    .op = OperatorType.Transpose,
                    .type = f32,
                    .params = .{
                        .inp_0 = "Input",
                        .inp_0_shape = .{ 2, 3 },
                        .out_0 = "Output",
                        .out_0_shape = .{ 3, 2 },
                        .perm = .{ 1, 0 },
                    },
                },
            },
        },
    ).make();

    model.process();

    try std.testing.expect(model.inner.tensors.Output.data[0] == (@as(f32, 1.0)));
    try std.testing.expect(model.inner.tensors.Output.data[1] == (@as(f32, 4.0)));
    try std.testing.expect(model.inner.tensors.Output.data[2] == (@as(f32, 2.0)));
    try std.testing.expect(model.inner.tensors.Output.data[3] == (@as(f32, 5.0)));
    try std.testing.expect(model.inner.tensors.Output.data[4] == (@as(f32, 3.0)));
    try std.testing.expect(model.inner.tensors.Output.data[5] == (@as(f32, 6.0)));
}
