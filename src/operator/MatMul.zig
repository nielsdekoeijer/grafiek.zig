const std = @import("std");
const utils = @import("../core/utils.zig");
const OperatorType = @import("../operator/OperatorType.zig").OperatorType;
const TensorType = @import("../tensor/TensorType.zig").TensorType;
const Make = @import("../core/Make.zig").Make;

pub fn MatMul(comptime Self: type, comptime _A: anytype, comptime _B: anytype, comptime _Y: anytype) type {
    comptime {
        return struct {
            pub fn process(self: *Self) void {
                const A = @TypeOf(@field(self.inner.tensors, _A));
                const B = @TypeOf(@field(self.inner.tensors, _B));
                const Y = @TypeOf(@field(self.inner.tensors, _Y));

                _ = A;
                _ = B;
                _ = Y;

            }
        };
    }
}

// test "operator.Sigmoid" {
//     var model = Make(
//         .{
//             .tensors = .{
//                 .{
//                     .name = "Input",
//                     .type = f32,
//                     .shape = .{ 2, 3 },
//                     .tensor = TensorType.Input,
//                     .params = .{
//                         .data = .{ -1.0, -2.0, -3.0, -4.0, -5.0, -6.0 },
//                     },
//                 },
//                 .{
//                     .name = "Output",
//                     .type = f32,
//                     .shape = .{ 2, 3 },
//                     .tensor = TensorType.Output,
//                     .params = {},
//                 },
//             },
//             .operators = .{
//                 .{
//                     .name = "MatMul",
//                     .op = OperatorType.MatMul,
//                     .type = f32,
//                     .params = .{
//                         .inp_0 = "Input",
//                         .inp_0_shape = .{ 2, 3 },
//                         .out_0 = "Output",
//                         .out_0_shape = .{ 2, 3 },
//                     },
//                 },
//             },
//         },
//     ).make();
//
//     model.process();
//
//     try std.testing.expect(model.inner.tensors.Output.data[0] == 1.0);
//     try std.testing.expect(model.inner.tensors.Output.data[1] == 2.0);
//     try std.testing.expect(model.inner.tensors.Output.data[2] == 3.0);
//     try std.testing.expect(model.inner.tensors.Output.data[3] == 4.0);
//     try std.testing.expect(model.inner.tensors.Output.data[4] == 5.0);
//     try std.testing.expect(model.inner.tensors.Output.data[5] == 6.0);
// }
