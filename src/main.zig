const std = @import("std");

pub const op = enum {
    Linear,
    Conv,
    Relu,
    MaxPool,
    Add,
    MatMul,
    Reshape,
};

pub fn Linear(comptime T: type) type {
    comptime {
        _ = T;
        return struct {};
    }
}

pub fn Conv(comptime T: type) type {
    comptime {
        _ = T;
        return struct {};
    }
}

pub fn Relu(comptime T: type) type {
    comptime {
        _ = T;
        return struct {};
    }
}

pub fn MaxPool(comptime T: type) type {
    comptime {
        _ = T;
        return struct {};
    }
}

pub fn Add(comptime T: type) type {
    comptime {
        _ = T;
        return struct {};
    }
}

pub fn MatMul(comptime T: type) type {
    comptime {
        _ = T;
        return struct {};
    }
}

pub fn Reshape(comptime T: type) type {
    comptime {
        _ = T;
        return struct {};
    }
}

pub fn Tensor(comptime T: type, comptime tensorShape: anytype) type {
    comptime {
        const tensorDim = tensorShape.len;
        var tensorLen = 1;
        for (tensorShape) |s| {
            tensorLen *= s;
        }

        return struct {
            const Self = @This();
            data: [tensorLen]T,

            pub fn make() Self {
                return Self {
                    .data = [_]T{0} ** tensorLen,
                };
            }

            inline fn get(this: Self, index: anytype) T {
                comptime var offset = 0;
                comptime {
                    for (0..tensorDim) |i| {
                        offset = offset * tensorShape[i] + index[i];
                    }
                }

                return this.data[offset];
            }

            inline fn dim() usize {
                comptime {
                    return tensorDim;
                }
            }

            inline fn shape() @TypeOf(tensorShape) {
                comptime {
                    return tensorShape;
                }
            }
        };
    }
}

pub fn MakeModel(comptime M: anytype) type {
    comptime {
        var TensorType: type = undefined;
        {
            var fields: [M.tensors.len]std.builtin.Type.StructField = undefined;
            for (0..M.tensors.len) |idx| {
                fields[idx] = std.builtin.Type.StructField{
                    .name = M.tensors[idx].name,
                    .type = Tensor(f32, M.tensors[idx].shape),
                    .default_value = &Tensor(f32, M.tensors[idx].shape).make(),
                    .is_comptime = false,
                    .alignment = 0,
                };
            }

            TensorType = @Type(.{
                .@"struct" = .{
                    .layout =  .auto,
                    .fields = fields[0..],
                    .decls = &[_]std.builtin.Type.Declaration{},
                    .is_tuple = false,
                }
            });
        }
        const tensorList = std.builtin.Type.StructField{
                    .name = "tensors",
                    .type = TensorType,
                    .default_value = &TensorType {},
                    .is_comptime = false,
                    .alignment = 0,
                };

        var OperationType: type = undefined;
        {
            var fields: [M.operations.len]std.builtin.Type.StructField = undefined;
            for (0..M.tensors.len) |idx| {

                const Op = switch(M.operations[idx].op) {
                    op.Add => Add(M.operations[idx].type),
                    op.Conv => Conv(M.operations[idx].type),
                    op.Relu => Relu(M.operations[idx].type),
                    op.MatMul => MatMul(M.operations[idx].type),
                    op.Linear => Linear(M.operations[idx].type),
                    op.MaxPool => MaxPool(M.operations[idx].type),
                    op.Reshape => Reshape(M.operations[idx].type),
                };

                fields[idx] = std.builtin.Type.StructField{
                    .name = M.operations[idx].name,
                    .type = Op,
                    .default_value = &Op {},
                    .is_comptime = false,
                    .alignment = 0,
                };
            }

            OperationType = @Type(.{
                .@"struct" = .{
                    .layout =  .auto,
                    .fields = fields[0..],
                    .decls = &[_]std.builtin.Type.Declaration{},
                    .is_tuple = false,
                }
            });
        }
        const operationList = std.builtin.Type.StructField{
                    .name = "operations",
                    .type = OperationType,
                    .default_value = &OperationType {},
                    .is_comptime = false,
                    .alignment = 0,
                };


        const fields: [2]std.builtin.Type.StructField = .{tensorList, operationList};
        return @Type(.{
            .@"struct" = .{
                .layout =  .auto,
                .fields = fields[0..],
                .decls = &[_]std.builtin.Type.Declaration{},
                .is_tuple = false,
            }
        });
    }
}

pub fn main() !void {
    const model = MakeModel(@import("model.zig").model) {};
    std.debug.print("{}\n", .{model});
}
