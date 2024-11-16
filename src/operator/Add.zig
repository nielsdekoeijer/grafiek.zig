const std = @import("std");

pub fn Add(comptime Self: type, comptime inp_a: anytype, comptime inp_b: anytype, comptime out: anytype) type {
    comptime {
        return struct {
            pub fn process(self: *Self) void {
                inline for (0..@TypeOf(@field(self.inner.tensors, out))._len()) |i| {
                    @field(self.inner.tensors, out).data[i] = @field(self.inner.tensors, inp_a).data[i] + @field(self.inner.tensors, inp_b).data[i];
                }
            }
        };
    }
}

const utils = @import("../core/utils.zig");
const OperatorType = @import("../operator/OperatorType.zig").OperatorType;
const TensorType = @import("../tensor/TensorType.zig").TensorType;
const Make = @import("../core/Make.zig").Make;

test "operator.Add" {
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
                    .name = "Constant",
                    .type = f32,
                    .shape = .{ 2, 3 },
                    .tensor = TensorType.Fixed,
                    .params = .{
                        .data = .{ 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 },
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
                    .name = "Add",
                    .op = OperatorType.Add,
                    .type = f32,
                    .params = .{
                        .inp_0 = "Input",
                        .inp_0_shape = .{ 2, 3 },
                        .inp_1 = "Constant",
                        .inp_1_shape = .{ 2, 3 },
                        .out_0 = "Output",
                        .out_0_shape = .{ 2, 3 },
                    },
                },
            },
        },
    ).make();

    model.process();

    try std.testing.expect(model.inner.tensors.Output.data[0] == 7.0);
    try std.testing.expect(model.inner.tensors.Output.data[1] == 7.0);
    try std.testing.expect(model.inner.tensors.Output.data[2] == 7.0);
    try std.testing.expect(model.inner.tensors.Output.data[3] == 7.0);
    try std.testing.expect(model.inner.tensors.Output.data[4] == 7.0);
    try std.testing.expect(model.inner.tensors.Output.data[5] == 7.0);
}
