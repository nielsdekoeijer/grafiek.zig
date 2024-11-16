const std = @import("std");

pub inline fn comptimeShapesEqual(comptime a: anytype, comptime b: anytype) bool {
    comptime {
        for (0..a.len) |i| {
            if (a[i] != b[i]) {
                return false;
            }
        }

        return true;
    }
}

pub inline fn comptimeDimEqual(comptime a: anytype, comptime b: anytype) bool {
    comptime {
        return a.len == b.len;
    }
}

pub inline fn comptimeAssert(comptime cond: bool) void {
    comptime {
        if (!cond) {
            @compileError("comptime assert failed");
        }
    }
}

// permutations
inline fn ComptimeIndexPermuteType(comptime indices: anytype, comptime perm: anytype) type {
    comptime {
        var fields: [perm.len]std.builtin.Type.StructField = undefined;
        for (0..perm.len) |idx| {
            fields[idx] = std.builtin.Type.StructField{
                .name = std.fmt.comptimePrint("{}", .{idx}),
                .type = @TypeOf(indices[perm[idx]]),
                .default_value = &indices[perm[idx]],
                .is_comptime = true,
                .alignment = 0,
            };
        }

        return @Type(
            .{
                .@"struct" = .{
                    .layout = .auto,
                    .fields = fields[0..],
                    .decls = &[_]std.builtin.Type.Declaration{},
                    .is_tuple = true,
                },
            },
        );
    }
}

pub inline fn comptimeIndexPermute(comptime indices: anytype, comptime perm: anytype) ComptimeIndexPermuteType(indices, perm) {
    comptime {
        comptimeAssert(comptimeDimEqual(indices, perm));
        return ComptimeIndexPermuteType(indices, perm){};
    }
}

// generates a list that, in order, iterates over an entire shape
inline fn ComptimeIndexAsShapeType(comptime shape: anytype, comptime index: comptime_int) type {
    comptime {
        var fields: [shape.len]std.builtin.Type.StructField = undefined;
        var remainingIndex = index;

        for (0..shape.len) |idx| {
            var stride = 1;
            for (idx + 1..shape.len) |jdx| {
                stride *= shape[jdx];
            }

            fields[idx] = std.builtin.Type.StructField{
                .name = std.fmt.comptimePrint("{}", .{idx}),
                .type = comptime_int,
                .default_value = &(remainingIndex / stride),
                .is_comptime = true,
                .alignment = 0,
            };

            remainingIndex %= stride;
        }

        return @Type(
            .{
                .@"struct" = .{
                    .layout = .auto,
                    .fields = fields[0..],
                    .decls = &[_]std.builtin.Type.Declaration{},
                    .is_tuple = true,
                },
            },
        );
    }
}

pub inline fn comptimeIndexAsShape(comptime shape: anytype, comptime index: usize) ComptimeIndexAsShapeType(shape, index) {
    comptime {
        return ComptimeIndexAsShapeType(shape, index){};
    }
}
