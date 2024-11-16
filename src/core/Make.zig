const std = @import("std");
const utils = @import("utils.zig");

const OperatorType = @import("../operator/OperatorType.zig").OperatorType;
const Tensor = @import("../tensor/Tensor.zig").Tensor;

const Abs = @import("../operator/Abs.zig").Abs;
const Acos = @import("../operator/Acos.zig").Acos;
const Acosh = @import("../operator/Acosh.zig").Acosh;
const Asin = @import("../operator/Asin.zig").Asin;
const Asinh = @import("../operator/Asinh.zig").Asinh;
const Atan = @import("../operator/Atan.zig").Atan;
const Atanh = @import("../operator/Atanh.zig").Atanh;
const Ceil = @import("../operator/Ceil.zig").Ceil;
const Add = @import("../operator/Add.zig").Add;
const Mul = @import("../operator/Mul.zig").Mul;
const Identity = @import("../operator/Identity.zig").Identity;
const Reshape = @import("../operator/Reshape.zig").Reshape;
const Transpose = @import("../operator/Transpose.zig").Transpose;
const Sigmoid = @import("../operator/Sigmoid.zig").Sigmoid;
const Tan = @import("../operator/Tan.zig").Tan;
const Tanh = @import("../operator/Tanh.zig").Tanh;
const Floor = @import("../operator/Floor.zig").Floor;

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
                            OperatorType.Mul => blk: {
                                utils.comptimeAssert(utils.comptimeShapesEqual(
                                    M.operators[idx].params.out_0_shape,
                                    M.operators[idx].params.inp_0_shape,
                                ));
                                utils.comptimeAssert(utils.comptimeShapesEqual(
                                    M.operators[idx].params.out_0_shape,
                                    M.operators[idx].params.inp_1_shape,
                                ));

                                break :blk Mul(
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
                            OperatorType.Reshape => blk: {
                                break :blk Reshape(
                                    @This(),
                                    M.operators[idx].params.inp_0,
                                    M.operators[idx].params.out_0,
                                );
                            },
                            OperatorType.Transpose => blk: {
                                break :blk Transpose(
                                    @This(),
                                    M.operators[idx].params.inp_0,
                                    M.operators[idx].params.out_0,
                                    M.operators[idx].params.perm,
                                );
                            },
                            OperatorType.Sigmoid => blk: {
                                utils.comptimeAssert(utils.comptimeShapesEqual(
                                    M.operators[idx].params.out_0_shape,
                                    M.operators[idx].params.inp_0_shape,
                                ));

                                break :blk Sigmoid(
                                    @This(),
                                    M.operators[idx].params.inp_0,
                                    M.operators[idx].params.out_0,
                                );
                            },
                            OperatorType.Tan => blk: {
                                utils.comptimeAssert(utils.comptimeShapesEqual(
                                    M.operators[idx].params.out_0_shape,
                                    M.operators[idx].params.inp_0_shape,
                                ));

                                break :blk Tan(
                                    @This(),
                                    M.operators[idx].params.inp_0,
                                    M.operators[idx].params.out_0,
                                );
                            },
                            OperatorType.Tanh => blk: {
                                utils.comptimeAssert(utils.comptimeShapesEqual(
                                    M.operators[idx].params.out_0_shape,
                                    M.operators[idx].params.inp_0_shape,
                                ));

                                break :blk Tanh(
                                    @This(),
                                    M.operators[idx].params.inp_0,
                                    M.operators[idx].params.out_0,
                                );
                            },
                            OperatorType.Abs => blk: {
                                utils.comptimeAssert(utils.comptimeShapesEqual(
                                    M.operators[idx].params.out_0_shape,
                                    M.operators[idx].params.inp_0_shape,
                                ));

                                break :blk Abs(
                                    @This(),
                                    M.operators[idx].params.inp_0,
                                    M.operators[idx].params.out_0,
                                );
                            },
                            OperatorType.Acos => blk: {
                                utils.comptimeAssert(utils.comptimeShapesEqual(
                                    M.operators[idx].params.out_0_shape,
                                    M.operators[idx].params.inp_0_shape,
                                ));

                                break :blk Acos(
                                    @This(),
                                    M.operators[idx].params.inp_0,
                                    M.operators[idx].params.out_0,
                                );
                            },
                            OperatorType.Acosh => blk: {
                                utils.comptimeAssert(utils.comptimeShapesEqual(
                                    M.operators[idx].params.out_0_shape,
                                    M.operators[idx].params.inp_0_shape,
                                ));

                                break :blk Acosh(
                                    @This(),
                                    M.operators[idx].params.inp_0,
                                    M.operators[idx].params.out_0,
                                );
                            },
                            OperatorType.Asin => blk: {
                                utils.comptimeAssert(utils.comptimeShapesEqual(
                                    M.operators[idx].params.out_0_shape,
                                    M.operators[idx].params.inp_0_shape,
                                ));

                                break :blk Asin(
                                    @This(),
                                    M.operators[idx].params.inp_0,
                                    M.operators[idx].params.out_0,
                                );
                            },
                            OperatorType.Asinh => blk: {
                                utils.comptimeAssert(utils.comptimeShapesEqual(
                                    M.operators[idx].params.out_0_shape,
                                    M.operators[idx].params.inp_0_shape,
                                ));

                                break :blk Asinh(
                                    @This(),
                                    M.operators[idx].params.inp_0,
                                    M.operators[idx].params.out_0,
                                );
                            },
                            OperatorType.Atan => blk: {
                                utils.comptimeAssert(utils.comptimeShapesEqual(
                                    M.operators[idx].params.out_0_shape,
                                    M.operators[idx].params.inp_0_shape,
                                ));

                                break :blk Atan(
                                    @This(),
                                    M.operators[idx].params.inp_0,
                                    M.operators[idx].params.out_0,
                                );
                            },
                            OperatorType.Atanh => blk: {
                                utils.comptimeAssert(utils.comptimeShapesEqual(
                                    M.operators[idx].params.out_0_shape,
                                    M.operators[idx].params.inp_0_shape,
                                ));

                                break :blk Atanh(
                                    @This(),
                                    M.operators[idx].params.inp_0,
                                    M.operators[idx].params.out_0,
                                );
                            },
                            OperatorType.Ceil => blk: {
                                utils.comptimeAssert(utils.comptimeShapesEqual(
                                    M.operators[idx].params.out_0_shape,
                                    M.operators[idx].params.inp_0_shape,
                                ));

                                break :blk Ceil(
                                    @This(),
                                    M.operators[idx].params.inp_0,
                                    M.operators[idx].params.out_0,
                                );
                            },
                            OperatorType.Floor => blk: {
                                utils.comptimeAssert(utils.comptimeShapesEqual(
                                    M.operators[idx].params.out_0_shape,
                                    M.operators[idx].params.inp_0_shape,
                                ));

                                break :blk Floor(
                                    @This(),
                                    M.operators[idx].params.inp_0,
                                    M.operators[idx].params.out_0,
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

            pub fn make() @This() {
                return @This(){
                    .inner = Make(M).InnerType{},
                };
            }

            pub fn process(self: *@This()) void {
                inline for (self.inner.operators) |op| {
                    @TypeOf(op).process(self);
                }
            }
        };
    }
}
