import onnx
import onnx.helper
import onnx.shape_inference
import numpy as np
from onnx import numpy_helper

# Load and infer shapes in the model
model = onnx.load('model.onnx')
model = onnx.shape_inference.infer_shapes(model)

# Maps for data type and operation mappings
onnx_to_zig_type = {
    onnx.TensorProto.FLOAT: 'f32',
    onnx.TensorProto.UINT8: 'u8',
    onnx.TensorProto.INT8: 'i8',
    onnx.TensorProto.UINT16: 'u16',
    onnx.TensorProto.INT16: 'i16',
    onnx.TensorProto.INT32: 'i32',
    onnx.TensorProto.INT64: 'i64',
    onnx.TensorProto.STRING: '[]u8',
    onnx.TensorProto.BOOL: 'bool',
    onnx.TensorProto.FLOAT16: 'f16',
    onnx.TensorProto.DOUBLE: 'f64',
    # Add other types as needed
}

op_type_to_ops = {
    'Conv': 'Conv',
    'Add': 'Add',
    'Relu': 'Relu',
    'MaxPool': 'MaxPool',
    'MatMul': 'MatMul',
    'Reshape': 'Reshape',
    'Identity': 'Identity',
    # Add other mappings as needed
}

# Collect tensors
tensors = {}

# Process initializers (weights and biases)
for initializer in model.graph.initializer:
    name = initializer.name
    data_type = initializer.data_type
    shape = [dim for dim in initializer.dims]
    zig_type = onnx_to_zig_type.get(data_type, 'unknown')
    data_array = numpy_helper.to_array(initializer).flatten()
    data_list = data_array.tolist()
    tensors[name] = {
        'name': name,
        'type': zig_type,
        'shape': shape,
        'tensor': 'Fixed',  # Adjusted according to new syntax
        'params': {
            'data': data_list  # Include the data as a list
        },
    }

# Process inputs
for input in model.graph.input:
    name = input.name
    if name in tensors:
        continue
    data_type = input.type.tensor_type.elem_type
    shape = [dim.dim_value if (dim.dim_value > 0) else -1 for dim in input.type.tensor_type.shape.dim]
    zig_type = onnx_to_zig_type.get(data_type, 'unknown')
    tensors[name] = {
        'name': name,
        'type': zig_type,
        'shape': shape,
        'tensor': 'Input',
        'params': {},
    }

# Process outputs
for output in model.graph.output:
    name = output.name
    if name in tensors:
        continue
    data_type = output.type.tensor_type.elem_type
    shape = [dim.dim_value if (dim.dim_value > 0) else -1 for dim in output.type.tensor_type.shape.dim]
    zig_type = onnx_to_zig_type.get(data_type, 'unknown')
    tensors[name] = {
        'name': name,
        'type': zig_type,
        'shape': shape,
        'tensor': 'Output',
        'params': {},
    }

# Process intermediate tensors
for value_info in model.graph.value_info:
    name = value_info.name
    if name in tensors:
        continue
    data_type = value_info.type.tensor_type.elem_type
    shape = [dim.dim_value if (dim.dim_value > 0) else -1 for dim in value_info.type.tensor_type.shape.dim]
    zig_type = onnx_to_zig_type.get(data_type, 'unknown')
    tensors[name] = {
        'name': name,
        'type': zig_type,
        'shape': shape,
        'tensor': 'Computed',
        'params': {},
    }

# Collect operators
operators = []

for node in model.graph.node:
    op_name = node.name if node.name else node.output[0]
    op_type = node.op_type
    inputs = node.input
    outputs = node.output
    attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
    # Convert attributes to a Zig-friendly format
    zig_attrs = {}
    for k, v in attrs.items():
        if isinstance(v, bytes):
            v = v.decode('utf-8')
        zig_attrs[k] = v
    # Map ONNX op_type to Zig OperatorType enum
    op_enum = op_type_to_ops.get(op_type, 'UnknownOp')
    operator = {
        'name': op_name,
        'op': op_enum,
        'type': 'f32',  # Assuming f32; adjust if necessary
        'params': zig_attrs,
        'inputs': inputs,
        'outputs': outputs,
    }
    operators.append(operator)

# Generate Zig code
zig_code = '''
const OperatorType = @import("OperatorType.zig").OperatorType;
const TensorType = @import("TensorType.zig").TensorType;
pub const model = .{
    .tensors = .{
'''

# Add tensors to Zig code
for tensor in tensors.values():
    zig_code += '        .{\n'
    zig_code += f'            .name = "{tensor["name"]}",\n'
    zig_code += f'            .type = {tensor["type"]},\n'
    shape_str = ', '.join(map(str, tensor['shape']))
    zig_code += f'            .shape = .{{ {shape_str} }},\n'
    zig_code += f'            .tensor = TensorType.{tensor["tensor"]},\n'
    # Include data in .params if available
    if 'data' in tensor['params']:
        data_list = tensor['params']['data']
        # Convert data list to Zig array string
        data_str = ', '.join(map(str, data_list))
        zig_code += f'            .params = .{{ .data = .{{ {data_str} }} }},\n'
    else:
        zig_code += f'            .params = {{}},\n'
    zig_code += '        },\n'

zig_code += '    },\n    .operators = .{\n'

# Add operators to Zig code
for op in operators:
    zig_code += '        .{\n'
    zig_code += f'            .name = "{op["name"]}",\n'
    zig_code += f'            .op = OperatorType.{op["op"]},\n'
    zig_code += f'            .type = {op["type"]},\n'
    # Construct params including inputs and outputs
    params_dict = op['params']
    # Add inputs and outputs to params
    for idx, inp in enumerate(op['inputs']):
        input_shape = tensors.get(inp, {}).get('shape', [])
        params_dict[f'inp_{idx}'] = inp
        params_dict[f'inp_{idx}_shape'] = input_shape
    for idx, out in enumerate(op['outputs']):
        output_shape = tensors.get(out, {}).get('shape', [])
        params_dict[f'out_{idx}'] = out
        params_dict[f'out_{idx}_shape'] = output_shape
    # Convert params dictionary to Zig format
    params_str = '.{'
    for k, v in params_dict.items():
        if isinstance(v, list):
            v_str = '.{' + ', '.join(map(str, v)) + '}'
        elif isinstance(v, str):
            v_str = f'"{v}"'
        else:
            v_str = str(v)
        params_str += f'.{k} = {v_str}, '
    params_str += '}'
    zig_code += f'            .params = {params_str},\n'
    zig_code += '        },\n'

zig_code += '    },\n};\n'

# Output the generated Zig code
print(zig_code)

