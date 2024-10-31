const OperatorType = @import("OperatorType.zig").OperatorType;
const TensorType = @import("TensorType.zig").TensorType;
pub const model = .{
    .tensors = .{
        .{
            .name = "Input",
            .type = f32,
            .shape = .{ 1, 5 },
            .tensor = TensorType.Input,
            .params = {},
        },
        .{
            .name = "Constant",
            .type = f32,
            .shape = .{ 1, 5 },
            .tensor = TensorType.Fixed,
            .params = {},
        },
        .{
            .name = "Result",
            .type = f32,
            .shape = .{ 1, 5 },
            .tensor = TensorType.Computed,
            .params = {},
        },
        .{
            .name = "Output",
            .type = f32,
            .shape = .{ 1, 5 },
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
                .inp_A = "Input",
                .inp_A_shape = .{ 1, 10 },
                .inp_B = "Constant",
                .inp_B_shape = .{ 1, 10 },
                .out = "Result",
                .out_shape = .{ 1, 10 },
            },
        },
        .{
            .name = "Identity",
            .op = OperatorType.Identity,
            .type = f32,
            .params = .{
                .inp = "Result",
                .inp_shape = .{ 1, 10 },
                .out = "Output",
                .out_shape = .{ 1, 10 },
            },
        },
    },
};
