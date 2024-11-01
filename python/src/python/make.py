import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

# Define tensor dimensions for a large tensor
tensor_shape = [100, 100]  # You can adjust these dimensions to make it even larger

# Create two large tensors filled with random float32 numbers
tensor_data_1 = np.random.rand(*tensor_shape).astype(np.float32)
tensor_data_2 = np.random.rand(*tensor_shape).astype(np.float32)

# Create initializers (constants) for the two input tensors
initializer_1 = numpy_helper.from_array(tensor_data_1, name='input_tensor_1')
initializer_2 = numpy_helper.from_array(tensor_data_2, name='input_tensor_2')

# Define input value infos (though we're using initializers, inputs can be optional)
input_1 = helper.make_tensor_value_info('input_tensor_1', TensorProto.FLOAT, tensor_shape)
input_2 = helper.make_tensor_value_info('input_tensor_2', TensorProto.FLOAT, tensor_shape)

# Define the output tensor
output = helper.make_tensor_value_info('output_tensor', TensorProto.FLOAT, tensor_shape)

# Create the Add node (the "giga big add function")
add_node = helper.make_node(
    'Add',                   # Operator name
    ['input_tensor_1', 'input_tensor_2'],  # Inputs
    ['output_tensor'],       # Outputs
    name='GigaBigAdd'        # Node name
)

# Create the graph
graph = helper.make_graph(
    [add_node],              # Nodes in the graph
    'GigaBigAddGraph',       # Graph name
    [input_1, input_2],      # Graph inputs
    [output],                # Graph outputs
    initializer=[initializer_1, initializer_2]  # Initializers (constants)
)

# Create the model
model = helper.make_model(graph, producer_name='onnx-example')
model = helper.make_model(graph, producer_name='onnx-example', opset_imports=[helper.make_operatorsetid("", 13)])

# Save the model to a file
onnx.save(model, 'model.onnx')

print("ONNX model 'model.onnx' has been generated.")

