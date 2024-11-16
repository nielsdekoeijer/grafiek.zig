const std = @import("std");
const Abs = @import("operator/Abs.zig");
const Acos = @import("operator/Acos.zig");
const Acosh = @import("operator/Acosh.zig");
const Asin = @import("operator/Asin.zig");
const Asinh = @import("operator/Asinh.zig");
const Atan = @import("operator/Atan.zig");
const Atanh = @import("operator/Atanh.zig");
const Ceil = @import("operator/Ceil.zig");
const Add = @import("operator/Add.zig");
const Mul = @import("operator/Mul.zig");
const MatMul = @import("operator/MatMul.zig");
const Tanh = @import("operator/Tanh.zig");
const Sigmoid = @import("operator/Sigmoid.zig");
const Transpose = @import("operator/Transpose.zig");
const Floor = @import("operator/Floor.zig");

const Make = @import("core/Make.zig").Make;

test {
    std.testing.refAllDecls(Abs);
    std.testing.refAllDecls(Acos);
    std.testing.refAllDecls(Acosh);
    std.testing.refAllDecls(Asin);
    std.testing.refAllDecls(Asinh);
    std.testing.refAllDecls(Atan);
    std.testing.refAllDecls(Atanh);
    std.testing.refAllDecls(Ceil);
    std.testing.refAllDecls(Add);
    std.testing.refAllDecls(Mul);
    std.testing.refAllDecls(MatMul);
    std.testing.refAllDecls(Tanh);
    std.testing.refAllDecls(Sigmoid);
    std.testing.refAllDecls(Transpose);
}
