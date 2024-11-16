const std = @import("std");
const utils = @import("../core/utils.zig");

pub fn Ceil(comptime Self: type, comptime inp: anytype, comptime out: anytype) type {
    comptime {
        return struct {
            pub fn process(self: *Self) void {
                inline for (0..@TypeOf(@field(self.inner.tensors, out))._len()) |i| {
                    @field(self.inner.tensors, out).data[i] = @ceil(@field(self.inner.tensors, inp).data[i]);
                }
            }
        };
    }
}

const OperatorType = @import("../operator/OperatorType.zig").OperatorType;
const TensorType = @import("../tensor/TensorType.zig").TensorType;
const Make = @import("../core/Make.zig").Make;

test "operator.Sigmoid" {
    var model = Make(
        .{
            .tensors = .{
                .{
                    .name = "Input",
                    .type = f32,
                    .shape = .{ 2, 3 },
                    .tensor = TensorType.Input,
                    .params = .{
                        .data = .{ 0.9, 1.8, 2.6, 3.1, 4.3, 5.2 },
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
                    .name = "Ceil",
                    .op = OperatorType.Ceil,
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

    try std.testing.expect(model.inner.tensors.Output.data[0] == 1.0);
    try std.testing.expect(model.inner.tensors.Output.data[1] == 2.0);
    try std.testing.expect(model.inner.tensors.Output.data[2] == 3.0);
    try std.testing.expect(model.inner.tensors.Output.data[3] == 4.0);
    try std.testing.expect(model.inner.tensors.Output.data[4] == 5.0);
    try std.testing.expect(model.inner.tensors.Output.data[5] == 6.0);
}
