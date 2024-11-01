const std = @import("std");

const OperatorType = @import("OperatorType.zig").OperatorType;
const TensorType = @import("TensorType.zig").TensorType;
const Tensor = @import("Tensor.zig").Tensor;
const utils = @import("utils.zig");

// MatMul
// Sigmoid
// Split
// Batch Normalization
// Squeeze
// Slice
// LSTM
// InstanceNormalization
// Concat
// Gather
// Transpose

pub fn Transpose(comptime Self: type, comptime inp: anytype, comptime out: anytype, comptime perm: anytype) type {
    comptime {
        return struct {
            pub fn process(self: *Self) void {
                inline for (0..@TypeOf(@field(self.inner.tensors, out))._len()) |i| {
                    const j = utils.comptimeShapeForeach(@TypeOf(@field(self.inner.tensors, out))._shape(), i);
                    @field(self.inner.tensors, out).set(utils.comptimeIndexPermute(j, perm), @field(self.inner.tensors, inp).get(j));
                }
            }
        };
    }
}

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

pub fn Tanh(comptime Self: type, comptime inp: anytype, comptime out: anytype) type {
    comptime {
        return struct {
            pub fn process(self: *Self) void {
                inline for (0..@TypeOf(@field(self.inner.tensors, out))._len()) |i| {
                    @field(self.inner.tensors, out).data[i] = std.math.tanh(@field(self.inner.tensors, inp).data[i]);
                }
            }
        };
    }
}

pub fn Mul(comptime Self: type, comptime inp_a: anytype, comptime inp_b: anytype, comptime out: anytype) type {
    comptime {
        return struct {
            pub fn process(self: *Self) void {
                inline for (0..@TypeOf(@field(self.inner.tensors, out))._len()) |i| {
                    @field(self.inner.tensors, out).data[i] = @field(self.inner.tensors, inp_a).data[i] * @field(self.inner.tensors, inp_b).data[i];
                }
            }
        };
    }
}

pub fn Identity(comptime Self: type, comptime inp: anytype, comptime out: anytype) type {
    comptime {
        return struct {
            pub fn process(self: *Self) void {
                inline for (0..@TypeOf(@field(self.inner.tensors, out))._len()) |i| {
                    @field(self.inner.tensors, out).data[i] = @field(self.inner.tensors, inp).data[i];
                }
            }
        };
    }
}

pub fn Reshape(comptime Self: type, comptime inp: anytype, comptime out: anytype) type {
    comptime {
        return struct {
            pub fn process(self: *Self) void {
                inline for (0..@TypeOf(@field(self.inner.tensors, out))._len()) |i| {
                    @field(self.inner.tensors, out).data[i] = @field(self.inner.tensors, inp).data[i];
                }
            }
        };
    }
}

pub fn Make(comptime M: anytype) type {
    comptime {
        return struct {
            const InnerType = section: {
                var tensorType: type = undefined;
                {
                    var fields: [M.tensors.len]std.builtin.Type.StructField = undefined;
                    for (0..M.tensors.len) |idx| {
                        if (@TypeOf(M.tensors[idx].params) != void) {
                            fields[idx] = std.builtin.Type.StructField{
                                .name = M.tensors[idx].name,
                                .type = Tensor(f32, M.tensors[idx].shape),
                                .default_value = &Tensor(f32, M.tensors[idx].shape).init(M.tensors[idx].params.data),
                                .is_comptime = false,
                                .alignment = 0,
                            };
                        } else {
                            fields[idx] = std.builtin.Type.StructField{
                                .name = M.tensors[idx].name,
                                .type = Tensor(f32, M.tensors[idx].shape),
                                .default_value = &Tensor(f32, M.tensors[idx].shape).default(),
                                .is_comptime = false,
                                .alignment = 0,
                            };
                        }
                    }

                    tensorType = @Type(.{
                        .@"struct" = .{
                            .layout = .auto,
                            .fields = fields[0..],
                            .decls = &[_]std.builtin.Type.Declaration{},
                            .is_tuple = false,
                        },
                    });
                }

                var operatorType: type = undefined;
                {
                    var fields: [M.operators.len]std.builtin.Type.StructField = undefined;
                    for (0..M.operators.len) |idx| {
                        const Op = switch (M.operators[idx].op) {
                            OperatorType.Add => blk: {
                                utils.comptimeAssert(utils.comptimeShapesEqual(
                                    M.operators[idx].params.out_0_shape,
                                    M.operators[idx].params.inp_0_shape,
                                ));
                                utils.comptimeAssert(utils.comptimeShapesEqual(
                                    M.operators[idx].params.out_0_shape,
                                    M.operators[idx].params.inp_1_shape,
                                ));

                                break :blk Add(
                                    @This(),
                                    M.operators[idx].params.inp_0,
                                    M.operators[idx].params.inp_1,
                                    M.operators[idx].params.out_0,
                                );
                            },
                            OperatorType.Identity => blk: {
                                utils.comptimeAssert(utils.comptimeShapesEqual(
                                    M.operators[idx].params.out_0_shape,
                                    M.operators[idx].params.inp_0_shape,
                                ));

                                break :blk Identity(
                                    @This(),
                                    M.operators[idx].params.inp_0,
                                    M.operators[idx].params.out_0,
                                );
                            },
                            OperatorType.Transpose => blk: {
                                break :blk Transpose (
                                    @This(),
                                    M.operators[idx].params.inp_0,
                                    M.operators[idx].params.out_0,
                                    M.operators[idx].params.perm,
                                );
                            },
                        };

                        fields[idx] = std.builtin.Type.StructField{
                            .name = std.fmt.comptimePrint("{}", .{idx}),
                            .type = Op,
                            .default_value = &Op{},
                            .is_comptime = false,
                            .alignment = 0,
                        };
                    }

                    operatorType = @Type(.{
                        .@"struct" = .{
                            .layout = .auto,
                            .fields = fields[0..],
                            .decls = &[_]std.builtin.Type.Declaration{},
                            .is_tuple = true,
                        },
                    });
                }

                const fields: [2]std.builtin.Type.StructField = .{
                    std.builtin.Type.StructField{
                        .name = "tensors",
                        .type = tensorType,
                        .default_value = &tensorType{},
                        .is_comptime = false,
                        .alignment = 0,
                    },
                    std.builtin.Type.StructField{
                        .name = "operators",
                        .type = operatorType,
                        .default_value = &operatorType{},
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

                break :section ExportedType;
            };

            inner: InnerType,

            pub fn process(self: *@This()) void {
                inline for (self.inner.operators) |op| {
                    @TypeOf(op).process(self);
                }
            }
        };
    }
}

pub fn main() !void {
    const Model = @import("simple.zig").model;
    var model = Make(Model){
        .inner = Make(Model).InnerType{},
    };

    model.inner.tensors.Input.data[0] = 1.0;
    model.process();

    std.debug.print("{}\n", .{model.inner.tensors.Input});
    std.debug.print("{}\n", .{model.inner.tensors.Constant});
    std.debug.print("{}\n", .{model.inner.tensors.Result0});
    std.debug.print("{}\n", .{model.inner.tensors.Result1});
    std.debug.print("{}\n", .{model.inner.tensors.Output});
}
