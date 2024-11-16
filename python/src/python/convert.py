import argparse
import onnx
import onnx.helper
import onnx.shape_inference
import numpy as np
from onnx import numpy_helper
import textwrap
import os

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Convert an ONNX model to Zig code.')
    parser.add_argument('model', help='Path to the input ONNX model file.')
    parser.add_argument('-o', '--output_dir', help='Directory to output the Zig model files.', default='.')
    args = parser.parse_args()

    # Load the ONNX model from the specified file
    model = onnx.load(args.model)

    # Perform shape inference to populate missing shape information in the model
    model = onnx.shape_inference.infer_shapes(model)

    # Mapping from ONNX tensor data types to Zig data types
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

    # Dictionary to hold all tensors in the model
    tensors = {}

    # Prepare the output directories
    model_name = os.path.splitext(os.path.basename(args.model))[0]
    output_model_dir = os.path.join(args.output_dir, model_name)
    weights_dir = os.path.join(output_model_dir, 'weights')

    os.makedirs(weights_dir, exist_ok=True)

    # Process initializers (usually weights and biases), marking them as 'Fixed' tensors
    for initializer in model.graph.initializer:
        name = initializer.name
        data_type = initializer.data_type
        shape = list(initializer.dims)
        zig_type = onnx_to_zig_type.get(data_type, 'unknown')

        # Convert the initializer data to a NumPy array
        data_array = numpy_helper.to_array(initializer)
        data_list = data_array.flatten().tolist()  # Flatten and convert to list

        # Write the data to a separate Zig file in the weights directory
        weight_file_path = os.path.join(weights_dir, f'{name}.zig')
        with open(weight_file_path, 'w') as weight_file:
            weight_data_str = ', '.join(map(str, data_list))
            weight_file_content = f'pub const {name} = .{{ {weight_data_str} }};\n'
            weight_file.write(weight_file_content)

        # Store the tensor information in the tensors dictionary
        tensors[name] = {
            'name': name,
            'type': zig_type,
            'shape': shape,
            'tensor': 'Fixed',  # Mark as a fixed tensor (i.e., constant weights)
            'params': {
                'data_file': f'weights/{name}.zig',  # Reference to the weight file
            },
        }

    # Process the model inputs, marking them as 'Input' tensors
    for input in model.graph.input:
        name = input.name
        if name in tensors:
            continue  # Skip if already processed as an initializer

        tensor_type = input.type.tensor_type
        data_type = tensor_type.elem_type

        # Extract the shape, handling unknown dimensions
        shape = []
        for dim in tensor_type.shape.dim:
            if dim.HasField('dim_value'):
                shape.append(dim.dim_value)
            else:
                shape.append(-1)  # Use -1 to represent unknown dimensions

        zig_type = onnx_to_zig_type.get(data_type, 'unknown')

        # Store the tensor information
        tensors[name] = {
            'name': name,
            'type': zig_type,
            'shape': shape,
            'tensor': 'Input',  # Mark as an input tensor
            'params': {},
        }

    # Process the model outputs, marking them as 'Output' tensors
    for output in model.graph.output:
        name = output.name
        if name in tensors:
            continue  # Skip if already processed

        tensor_type = output.type.tensor_type
        data_type = tensor_type.elem_type

        # Extract the shape, handling unknown dimensions
        shape = []
        for dim in tensor_type.shape.dim:
            if dim.HasField('dim_value'):
                shape.append(dim.dim_value)
            else:
                shape.append(-1)  # Use -1 to represent unknown dimensions

        zig_type = onnx_to_zig_type.get(data_type, 'unknown')

        # Store the tensor information
        tensors[name] = {
            'name': name,
            'type': zig_type,
            'shape': shape,
            'tensor': 'Output',  # Mark as an output tensor
            'params': {},
        }

    # Process intermediate tensors (value_info), marking them as 'Computed' tensors
    for value_info in model.graph.value_info:
        name = value_info.name
        if name in tensors:
            continue  # Skip if already processed

        tensor_type = value_info.type.tensor_type
        data_type = tensor_type.elem_type

        # Extract the shape, handling unknown dimensions
        shape = []
        for dim in tensor_type.shape.dim:
            if dim.HasField('dim_value'):
                shape.append(dim.dim_value)
            else:
                shape.append(-1)  # Use -1 to represent unknown dimensions

        zig_type = onnx_to_zig_type.get(data_type, 'unknown')

        # Store the tensor information
        tensors[name] = {
            'name': name,
            'type': zig_type,
            'shape': shape,
            'tensor': 'Computed',  # Mark as a computed tensor (intermediate results)
            'params': {},
        }

    # List to hold all operators (nodes) in the model
    operators = []

    # Process each node (operator) in the model graph
    for node in model.graph.node:
        # Use the node's name if available; otherwise, use the first output as the name
        op_name = node.name if node.name else node.output[0]
        op_type = node.op_type  # The type of the operation (e.g., 'Conv', 'Add')
        inputs = node.input  # List of input tensor names
        outputs = node.output  # List of output tensor names

        # Collect all attributes of the node into a dictionary
        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}

        # Since we want to support all ops and use ONNX naming conventions,
        # we can use the op_type directly as the Zig OperatorType enum
        op_enum = op_type

        # Collect parameters, including attributes, inputs, outputs, and their shapes and types
        params = {}

        # Include all attributes in the parameters
        for k, v in attrs.items():
            if isinstance(v, bytes):
                v = v.decode('utf-8')  # Decode bytes to string if necessary
            params[k] = v

        # Include inputs, their shapes, and types in the parameters
        for idx, inp in enumerate(inputs):
            params[f'inp_{idx}'] = inp
            inp_shape = tensors.get(inp, {}).get('shape', [])
            inp_type = tensors.get(inp, {}).get('type', 'unknown')
            params[f'inp_{idx}_shape'] = inp_shape
            params[f'inp_{idx}_type'] = inp_type

        # Include outputs, their shapes, and types in the parameters
        for idx, out in enumerate(outputs):
            params[f'out_{idx}'] = out
            out_shape = tensors.get(out, {}).get('shape', [])
            out_type = tensors.get(out, {}).get('type', 'unknown')
            params[f'out_{idx}_shape'] = out_shape
            params[f'out_{idx}_type'] = out_type

        # Create the operator entry
        operator = {
            'name': op_name,
            'op': op_enum,
            'type': 'f32',  # Assuming f32; adjust if necessary
            'params': params,
        }

        # Add the operator to the list
        operators.append(operator)

    # Start building the Zig code as a string
    zig_code = textwrap.dedent('''\
        const OperatorType = @import("OperatorType.zig").OperatorType;
        const TensorType = @import("TensorType.zig").TensorType;
        pub const model = .{
            .tensors = .{
    ''')

    # Function to format a tensor entry into Zig code
    def format_tensor(tensor):
        lines = []
        lines.append('    .{')
        lines.append(f'        .name = "{tensor["name"]}",')
        lines.append(f'        .type = {tensor["type"]},')
        shape_str = ', '.join(map(str, tensor['shape']))
        lines.append(f'        .shape = .{{ {shape_str} }},')
        lines.append(f'        .tensor = TensorType.{tensor["tensor"]},')

        # For 'Fixed' tensors, import the data from the separate Zig file
        if tensor['tensor'] == 'Fixed':
            data_import = f'@import("{tensor["params"]["data_file"]}")'
            lines.append(f'        .params = .{{ .data = {data_import} }},')
        else:
            lines.append('        .params = {},')  # Empty parameters if no data

        lines.append('    },')
        return '\n'.join(lines)

    # Add each tensor to the Zig code
    for tensor in tensors.values():
        zig_code += format_tensor(tensor) + '\n'

    # Close the tensors array and start the operators array
    zig_code += '},\n    .operators = .{\n'

    # Function to format an operator entry into Zig code
    def format_operator(op):
        lines = []
        lines.append('    .{')
        lines.append(f'        .name = "{op["name"]}",')
        lines.append(f'        .op = OperatorType.{op["op"]},')
        lines.append(f'        .type = {op["type"]},')

        # Format the parameters dictionary into Zig syntax
        params_lines = []
        params_lines.append('.{')
        for k, v in op['params'].items():
            if isinstance(v, list):
                # Format lists (e.g., shapes) as Zig arrays
                v_str = '.{' + ', '.join(map(str, v)) + '}'
            elif isinstance(v, str):
                # Enclose strings in double quotes
                v_str = f'"{v}"'
            elif isinstance(v, float):
                # Format floats with precision
                v_str = f'{v:.6f}'
            else:
                v_str = str(v)
            params_lines.append(f'    .{k} = {v_str},')
        params_lines.append('}')
        params_str = '\n'.join(params_lines)
        lines.append(f'        .params = {params_str},')

        lines.append('    },')
        return '\n'.join(lines)

    # Add each operator to the Zig code
    for op in operators:
        zig_code += format_operator(op) + '\n'

    # Close the operators array and the model definition
    zig_code += '},\n};\n'

    # Write the main model.zig file
    model_file_path = os.path.join(output_model_dir, 'model.zig')
    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
    with open(model_file_path, 'w') as f:
        f.write(zig_code)

if __name__ == '__main__':
    main()

