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
