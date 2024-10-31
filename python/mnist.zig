const ops = @import("main.zig").ops;
const ten = @import("main.zig").ten;
pub const model = .{
    .tensors = .{
        .{
            .name = "Input3",
            .type = f32,
            .shape = .{ 1, 1, 28, 28 },
            .ten = ten.Input,
            .params = {},
        },
        .{
            .name = "Convolution28_Output_0",
            .type = f32,
            .shape = .{ 1, 8, 28, 28 },
            .ten = ten.Computed,
            .params = {},
        },
        .{
            .name = "Plus30_Output_0",
            .type = f32,
            .shape = .{ 1, 8, 28, 28 },
            .ten = ten.Computed,
            .params = {},
        },
        .{
            .name = "ReLU32_Output_0",
            .type = f32,
            .shape = .{ 1, 8, 28, 28 },
            .ten = ten.Computed,
            .params = {},
        },
        .{
            .name = "Pooling66_Output_0",
            .type = f32,
            .shape = .{ 1, 8, 14, 14 },
            .ten = ten.Computed,
            .params = {},
        },
        .{
            .name = "Convolution110_Output_0",
            .type = f32,
            .shape = .{ 1, 16, 14, 14 },
            .ten = ten.Computed,
            .params = {},
        },
        .{
            .name = "Plus112_Output_0",
            .type = f32,
            .shape = .{ 1, 16, 14, 14 },
            .ten = ten.Computed,
            .params = {},
        },
        .{
            .name = "ReLU114_Output_0",
            .type = f32,
            .shape = .{ 1, 16, 14, 14 },
            .ten = ten.Computed,
            .params = {},
        },
        .{
            .name = "Pooling160_Output_0_reshape0",
            .type = f32,
            .shape = .{ 1, 256 },
            .ten = ten.Computed,
            .params = {},
        },
        .{
            .name = "Parameter193_reshape1",
            .type = f32,
            .shape = .{ 256, 10 },
            .ten = ten.Computed,
            .params = {},
        },
        .{
            .name = "Times212_Output_0",
            .type = f32,
            .shape = .{ 1, 10 },
            .ten = ten.Computed,
            .params = {},
        },
        .{
            .name = "Output",
            .type = f32,
            .shape = .{ 1, 10 },
            .ten = ten.Output,
            .params = {},
        },
    },
    .operations = .{
        .{
            .name = "Convolution28",
            .op = ops.Conv,
            .type = f32,
            .params = .{
                .auto_pad = "NOTSET",
                .dilations = .{ 1, 1 },
                .group = 1,
                .kernel_shape = .{ 1, 1 },
                .pads = .{ 1, 1 },
                .strides = .{ 1, 1 },
                .X = .{ 1, 1 },
                .X_shape = .{ 1, 1 },
                .W = .{ 1, 1 },
                .W_shape = .{ 1, 1 },
            },
        },
        .{
            .name = "Plus30",
            .op = ops.Add,
            .type = f32,
            .params = .{
                .inp_A = "Input3",
                .inp_A_shape = .{ 1, 1, 28, 28 },
                .inp_B = "Input3",
                .inp_B_shape = .{ 1, 1, 28, 28 },
                .out = "Input3",
                .out_shape = .{ 1, 1, 28, 28 },
            },
        },
        .{
            .name = "ReLU32",
            .op = ops.Relu,
            .type = f32,
            .params = .{
                .inp = .{},
                .out = .{},
            },
        },
        .{
            .name = "Pooling66",
            .op = ops.MaxPool,
            .type = f32,
            .params = .{},
        },
        .{
            .name = "Convolution110",
            .op = ops.MaxPool,
            .type = f32,
            .params = .{},
        },
        .{
            .name = "Plus112",
            .op = ops.Add,
            .type = f32,
            .params = .{
                .inp_A = "Input3",
                .inp_A_shape = .{ 1, 1, 28, 28 },
                .inp_B = "Input3",
                .inp_B_shape = .{ 1, 1, 28, 28 },
                .out = "Input3",
                .out_shape = .{ 1, 1, 28, 28 },
            },
        },
        .{
            .name = "ReLU114",
            .op = ops.Relu,
            .type = f32,
            .params = .{},
        },
        .{
            .name = "Pooling160",
            .op = ops.MaxPool,
            .type = f32,
            .params = .{},
        },
        .{
            .name = "Times212_reshape0",
            .op = ops.MaxPool,
            .type = f32,
            .params = .{},
        },
        .{
            .name = "Times212_reshape1",
            .op = ops.Reshape,
            .type = f32,
            .params = .{},
        },
        .{
            .name = "Times212",
            .op = ops.MatMul,
            .type = f32,
            .params = .{},
        },
        .{
            .name = "Plus214",
            .op = ops.Add,
            .type = f32,
            .params = .{
                .inp_A = "Input3",
                .inp_A_shape = .{ 1, 1, 28, 28 },
                .inp_B = "Input3",
                .inp_B_shape = .{ 1, 1, 28, 28 },
                .out = "Input3",
                .out_shape = .{ 1, 1, 28, 28 },
            },
        },
    },
};