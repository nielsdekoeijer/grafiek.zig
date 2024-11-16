const OperatorType = @import("../operator/OperatorType.zig").OperatorType;
const TensorType = @import("../tensor/TensorType.zig").TensorType;
pub const model = .{
    .tensors = .{
        .{
            .name = "Input",
            .type = f32,
            .shape = .{ 2, 3 },
            .tensor = TensorType.Input,
            .params = {},
        },
        .{
            .name = "Constant",
            .type = f32,
            .shape = .{ 2, 3 },
            .tensor = TensorType.Fixed,
            .params = {},
        },
        .{
            .name = "Result0",
            .type = f32,
            .shape = .{ 2, 3 },
            .tensor = TensorType.Computed,
            .params = {},
        },
        .{
            .name = "Result1",
            .type = f32,
            .shape = .{ 3, 2 },
            .tensor = TensorType.Computed,
            .params = {},
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
                .out_0 = "Result0",
                .out_0_shape = .{ 2, 3 },
            },
        },
        .{
            .name = "Transpose",
            .op = OperatorType.Transpose,
            .type = f32,
            .params = .{
                .inp_0 = "Result0",
                .inp_0_shape = .{ 2, 3 },
                .out_0 = "Result1",
                .out_0_shape = .{ 3, 2 },
                .perm = .{ 1, 0 },
            },
        },
        .{
            .name = "Identity",
            .op = OperatorType.Identity,
            .type = f32,
            .params = .{
                .inp_0 = "Result1",
                .inp_0_shape = .{ 3, 2 },
                .out_0 = "Output",
                .out_0_shape = .{ 3, 2 },
            },
        },
    },
};
