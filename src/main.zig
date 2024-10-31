const std = @import("std");

const OperatorType = @import("OperatorType.zig").OperatorType;
const TensorType = @import("TensorType.zig").TensorType;
const Tensor = @import("Tensor.zig").Tensor;
const utils = @import("utils.zig");

pub fn Add(comptime Self: type, comptime inp_a: anytype, comptime inp_b: anytype, out: anytype) type {
    comptime {
        return struct {
            pub fn process(self: *Self) void {
                for (0..@TypeOf(@field(self.inner.tensors, out))._len()) |i| {
                    @field(self.inner.tensors, out).data[i] = @field(self.inner.tensors, inp_a).data[i] + @field(self.inner.tensors, inp_b).data[i];
                }
            }
        };
    }
}

pub fn Identity(comptime Self: type, comptime inp: anytype, out: anytype) type {
    comptime {
        return struct {
            pub fn process(self: *Self) void {
                for (0..@TypeOf(@field(self.inner.tensors, out))._len()) |i| {
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
                        fields[idx] = std.builtin.Type.StructField{
                            .name = M.tensors[idx].name,
                            .type = Tensor(f32, M.tensors[idx].shape),
                            .default_value = &Tensor(f32, M.tensors[idx].shape).default(),
                            .is_comptime = false,
                            .alignment = 0,
                        };

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
                                    M.operators[idx].params.out_shape,
                                    M.operators[idx].params.inp_A_shape,
                                ));
                                utils.comptimeAssert(utils.comptimeShapesEqual(
                                    M.operators[idx].params.out_shape,
                                    M.operators[idx].params.inp_B_shape,
                                ));

                                break :blk Add(
                                    @This(),
                                    M.operators[idx].params.inp_A,
                                    M.operators[idx].params.inp_B,
                                    M.operators[idx].params.out,
                                );
                            },
                            OperatorType.Identity => blk: {
                                utils.comptimeAssert(utils.comptimeShapesEqual(
                                    M.operators[idx].params.out_shape,
                                    M.operators[idx].params.inp_shape,
                                ));

                                break :blk Identity(
                                    @This(),
                                    M.operators[idx].params.inp,
                                    M.operators[idx].params.out,
                                );
                            },
                        };

                        fields[idx] = std.builtin.Type.StructField{
                            .name = M.operators[idx].name,
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
                            .is_tuple = false,
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
        };
    }
}

pub fn main() !void {
    const Model = @import("simple.zig").model;
    var model = Make(Model) {
        .inner = Make(Model).InnerType {},
    };

    var I0: f32 = undefined;
    const P0: *volatile f32 = @volatileCast(&I0);
    P0.* = 1.0;
    var I1: f32 = undefined;
    const P1: *volatile f32 = @volatileCast(&I1);
    P1.* = 2.0;

    model.inner.tensors.Input.set(.{0,0}, P0.*);
    model.inner.tensors.Input.set(.{0,1}, P1.*);

    model.inner.tensors.Constant.set(.{0,0}, 3.0);
    model.inner.tensors.Constant.set(.{0,1}, 4.0);

    @TypeOf(model.inner.operators.Add).process(&model);
    @TypeOf(model.inner.operators.Identity).process(&model);
    std.debug.print("{}\n", .{model.inner.tensors.Output.get(.{0,0})});
    std.debug.print("{}\n", .{model.inner.tensors.Output.get(.{0,1})});
}
