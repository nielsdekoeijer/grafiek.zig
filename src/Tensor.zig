const std = @import("std");

/// Tensor is a compile-time wrapper around a fixed-length array for representing
/// multi-dimensional data structures. This struct is initialized with a specified
/// element type and tensor shape.
///
/// Example usage:
/// ```zig
/// const T = Tensor(f32, .{3, 3});
/// const my_tensor = T.default();
/// ```
///
/// - `T` is the element type of the tensor.
/// - `tensorShape` defines the shape (dimensions) of the tensor.
pub fn Tensor(comptime T: type, comptime tensorShape: anytype) type {
    comptime {
        return struct {
            /// Self is a reference to the current type instance of the tensor.
            const Self = @This();

            /// `data` is a flat array storing the tensor's elements.
            data: [Self._len()]T,

            /// Initializes the tensor with a default value of `0` for each element.
            /// Returns a new tensor instance with all values set to zero.
            pub fn default() Self {
                return Self{
                    .data = [_]T{0} ** Self._len(),
                };
            }

            /// Retrieves the tensor element at the given multi-dimensional `index`.
            ///
            /// - `index`: an array representing the position within each dimension.
            /// - Returns the element of type `T` at the specified index.
            pub inline fn get(this: Self, index: anytype) T {
                comptime var offset = 0;
                comptime {
                    for (0..Self._dim()) |i| {
                        offset = offset * tensorShape[i] + index[i];
                    }
                }

                return this.data[offset];
            }

            /// Sets the tensor element at the given multi-dimensional `index`.
            ///
            /// - `index`: an array representing the position within each dimension.
            /// - `value`: the element of type `T` to set at the specified index.
            pub inline fn set(this: *Self, index: anytype, value: T) void {
                comptime var offset = 0;
                comptime {
                    for (0..Self._len()) |i| {
                        offset = offset * tensorShape[i] + index[i];
                    }
                }

                this.data[offset] = value;
            }

            /// Returns the total number of elements in the tensor.
            /// Computed as the product of each dimension size in `tensorShape`.
            pub inline fn len(comptime _: Self) usize {
                comptime {
                    return Self._len();
                }
            }

            /// Helper
            /// TODO(NIELS): can be better?
            pub inline fn _len() usize {
                comptime {
                    var tensorLen = 1;
                    for (tensorShape) |s| {
                        tensorLen *= s;
                    }
                    return tensorLen;
                }
            }

            /// Returns the number of dimensions of the tensor.
            /// Equivalent to the length of `tensorShape`.
            pub inline fn dim(comptime _: Self) usize {
                comptime {
                    return Self._dim();
                }
            }

            /// Helper
            /// TODO(NIELS): can be better?
            pub inline fn _dim() usize {
                comptime {
                    return tensorShape.len;
                }
            }

            /// Returns the shape of the tensor, which is an array defining
            /// the size of each dimension.
            pub inline fn shape(comptime _: Self) @TypeOf(tensorShape) {
                comptime {
                    return Self._shape();
                }
            }

            /// Helper
            /// TODO(NIELS): can be better?
            pub inline fn _shape(comptime _: Self) @TypeOf(tensorShape) {
                comptime {
                    return tensorShape;
                }
            }
        };
    }
}
