const op = @import("main.zig").op;
pub const model = .{
    .tensors = .{
        .{
            .name = "Input3",
            .type = f32,
            .shape = .{1, 1, 28, 28},
        },
        .{
            .name = "Convolution28_Output_0",
            .type = f32,
            .shape = .{1, 8, 28, 28},
        },
        .{
            .name = "Plus30_Output_0",
            .type = f32,
            .shape = .{1, 8, 28, 28},
        },
        .{
            .name = "ReLU32_Output_0",
            .type = f32,
            .shape = .{1, 8, 28, 28},
        },
        .{
            .name = "Pooling66_Output_0",
            .type = f32,
            .shape = .{1, 8, 14, 14},
        },
        .{
            .name = "Convolution110_Output_0",
            .type = f32,
            .shape = .{1, 16, 14, 14},
        },
        .{
            .name = "Plus112_Output_0",
            .type = f32,
            .shape = .{1, 16, 14, 14},
        },
        .{
            .name = "ReLU114_Output_0",
            .type = f32,
            .shape = .{1, 16, 14, 14},
        },
        .{
            .name = "Pooling160_Output_0_reshape0",
            .type = f32,
            .shape = .{1, 256},
        },
        .{
            .name = "Parameter193_reshape1",
            .type = f32,
            .shape = .{256, 10},
        },
        .{
            .name = "Times212_Output_0",
            .type = f32,
            .shape = .{1, 10},
        },
        .{
            .name = "Output",
            .type = f32,
            .shape = .{1, 10},
        },
    },
    .operations = .{
        .{
            .name = "Convolution28",
            .op = op.Conv,
            .type = f32,
            .params = .{
            }
        },
        .{
            .name = "Plus30",
            .op = op.Add,
            .type = f32,
            .params = .{
            }
        },
        .{
            .name = "ReLU32",
            .op = op.Relu,
            .type = f32,
            .params = .{
            }
        },
        .{
            .name = "Pooling66",
            .op = op.MaxPool,
            .type = f32,
            .params = .{
            }
        },
        .{
            .name = "Convolution110",
            .op = op.MaxPool,
            .type = f32,
            .params = .{
            }
        },
        .{
            .name = "Plus112",
            .op = op.Add,
            .type = f32,
            .params = .{
            }
        },
        .{
            .name = "ReLU114",
            .op = op.Relu,
            .type = f32,
            .params = .{
            }
        },
        .{
            .name = "Pooling160",
            .op = op.MaxPool,
            .type = f32,
            .params = .{
            }
        },
        .{
            .name = "Times212_reshape0",
            .op = op.MaxPool,
            .type = f32,
            .params = .{
            }
        },
        .{
            .name = "Times212_reshape1",
            .op = op.Reshape,
            .type = f32,
            .params = .{
            }
        },
        .{
            .name = "Times212",
            .op = op.MatMul,
            .type = f32,
            .params = .{
            }
        },
        .{
            .name = "Plus214",
            .op = op.Add,
            .type = f32,
            .params = .{
            }
        },
    }
};
