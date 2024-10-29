const std = @import("std");

pub const ops = enum {
    Linear,
    Conv,
    Relu,
    MaxPool,
    Add,
    MatMul,
    Reshape,
};

pub const ten = enum {
    Input,
    Output,
    Fixed,
    Computed,
};

pub fn Linear(comptime T: type) type {
    comptime {
        _ = T;
        return struct {
            const Self = @This();
            fn default() Self {
                return Self{};
            }
        };
    }
}

pub fn Conv(comptime T: type) type {
    comptime {
        _ = T;
        return struct {
            const Self = @This();
            fn default() Self {
                return Self{};
            }
        };
    }
}

pub fn Relu(comptime T: type) type {
    comptime {
        _ = T;
        return struct {
            const Self = @This();
            fn default() Self {
                return Self{};
            }
        };
    }
}

pub fn MaxPool(comptime T: type) type {
    comptime {
        _ = T;
        return struct {
            const Self = @This();
            fn default() Self {
                return Self{};
            }
        };
    }
}

pub fn comptimeShapesEqual(comptime a: anytype, comptime b: anytype) bool {
    comptime {
        for (0..a.len) |i| {
            if (a[i] != b[i]) {
                return false;
            }
        }

        return true;
    }
}

pub fn comptimeAssert(comptime cond: bool) void {
    if (!cond) {
        @compileError("comptime assert failed");
    }
}

pub fn Add(comptime T: type, comptime shape: anytype) type {
    comptime {
        const Type = struct {
            const Self = @This();

            fn default() Self {
                return Self{
                    .inp = undefined,
                    .out = undefined,
                };
            }

            fn make(inp: [2]*Tensor(T, shape), out: [1]*Tensor(T, shape)) Self {
                return Self{
                    .inp = inp,
                    .out = out,
                };
            }

            fn process(self: *Self) void {
                for (0..self.out.data.len) |i| {
                    self.out[0][i] = self.inp[0][i] + self.inp[1][i];
                }
            }

            inp: [2]*Tensor(T, shape),
            out: [1]*Tensor(T, shape),
        };

        return Type;
    }
}

pub fn MatMul(comptime T: type) type {
    comptime {
        _ = T;
        return struct {
            const Self = @This();
            fn default() Self {
                return Self{};
            }
        };
    }
}

pub fn Reshape(comptime T: type) type {
    comptime {
        _ = T;
        return struct {
            const Self = @This();
            fn default() Self {
                return Self{};
            }
        };
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

            pub fn default() Self {
                return Self{
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

pub fn MakeModelInnerType(comptime M: anytype) type {
    comptime {
        var TensorType: type = undefined;
        {
            var fields: [M.tensors.len]std.builtin.Type.StructField = undefined;
            for (0..M.tensors.len) |idx| {
                fields[idx] = std.builtin.Type.StructField{
                    .name = M.tensors[idx].name,
                    .type = Tensor(f32, M.tensors[idx].shape),
                    .default_value = &Tensor(f32, M.tensors[idx].shape).default(),
                    .is_comptime = false,
                    .alignment = 0,
                };
            }

            TensorType = @Type(.{
                .@"struct" = .{
                    .layout = .auto,
                    .fields = fields[0..],
                    .decls = &[_]std.builtin.Type.Declaration{},
                    .is_tuple = false,
                },
            });
        }

        var OperationType: type = undefined;
        {
            var fields: [M.operations.len]std.builtin.Type.StructField = undefined;
            for (0..M.operations.len) |idx| {
                const Op = switch (M.operations[idx].op) {
                    ops.Add => blk: {
                        comptimeAssert(comptimeShapesEqual(
                            M.operations[idx].params.out_shape,
                            M.operations[idx].params.inp_A_shape,
                        ));
                        comptimeAssert(comptimeShapesEqual(
                            M.operations[idx].params.out_shape,
                            M.operations[idx].params.inp_B_shape,
                        ));

                        break :blk Add(M.operations[idx].type, M.operations[idx].params.out_shape);
                    },
                    ops.Conv => Conv(M.operations[idx].type),
                    ops.Relu => Relu(M.operations[idx].type),
                    ops.MatMul => MatMul(M.operations[idx].type),
                    ops.Linear => Linear(M.operations[idx].type),
                    ops.MaxPool => MaxPool(M.operations[idx].type),
                    ops.Reshape => Reshape(M.operations[idx].type),
                };

                fields[idx] = std.builtin.Type.StructField{
                    .name = M.operations[idx].name,
                    .type = Op,
                    .default_value = &Op.default(),
                    .is_comptime = false,
                    .alignment = 0,
                };
            }

            OperationType = @Type(.{
                .@"struct" = .{
                    .layout = .auto,
                    .fields = fields[0..],
                    .decls = &[_]std.builtin.Type.Declaration{},
                    .is_tuple = false,
                },
            });
        }

        const fields: [2]std.builtin.Type.StructField = .{
            std.builtin.Type.StructField{
                .name = "tensors",
                .type = TensorType,
                .default_value = &TensorType{},
                .is_comptime = false,
                .alignment = 0,
            },
            std.builtin.Type.StructField{
                .name = "operations",
                .type = OperationType,
                .default_value = &OperationType{},
                .is_comptime = false,
                .alignment = 0,
            },
        };

        const ExportedType = @Type(
            .{
                .@"struct" = .{
                    .layout = .auto,
                    .fields = fields[0..],
                    .decls = &[_]std.builtin.Type.Declaration{},
                    .is_tuple = false,
                },
            },
        );

        return ExportedType;
    }
}

pub fn MakeModelType(comptime M: anytype) type {
    comptime {
        const T = MakeModelInnerType(M);
        return struct {
            const Self = @This();
            inner: T,

            pub fn make() Self {
                const inner = comptime blk: {
                    var inner = T{};
                    for (0..M.operations.len) |idx| {
                        switch (M.operations[idx].op) {
                            ops.Add => {
                                @field(inner.operations, M.operations[idx].name) = @TypeOf(
                                    @field(inner.operations, M.operations[idx].name),
                                ).make(
                                    .{
                                        &@field(inner.tensors, M.operations[idx].params.inp_A),
                                        &@field(inner.tensors, M.operations[idx].params.inp_B),
                                    },
                                    .{
                                        &@field(inner.tensors, M.operations[idx].params.out),
                                    },
                                );
                            },
                            else => {},
                        }
                    }
                    break :blk inner;
                };

                return Self{ .inner = inner };
            }
        };
    }
}

pub fn main() !void {
    const ModelConf = @import("model.zig").model;
    const ModelType = MakeModelType(ModelConf);
    const model = ModelType.make();
    std.debug.print("{}\n", .{model});
}
