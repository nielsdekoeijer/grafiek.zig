#!/bin/python3
import onnxruntime as ort
import numpy as np
import os
import time
import argparse

N = 10000

onnx_models = ["model.onnx"]

# Filter only ONNX models (by file extension)
duration_map = {}
print(f"Running {len(onnx_models)} models...")
for model in onnx_models:
    print(f"\n")
    print(f"Loading {model}")

    # create session with ort
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1 
    session_options.inter_op_num_threads = 1

    # Create session with options
    session = ort.InferenceSession(f"{model}", sess_options=session_options, providers=["CPUExecutionProvider"])

    # get names
    inputs = {}
    for inp in session.get_inputs():
        print(f"- Found input with name {inp.name}")
        print(f"- Found input with shape {inp.shape}")
        if inp.type == "tensor(float)":
            print(f"- Setting inputs to zero...")
            inputs[inp.name] = np.zeros(inp.shape, dtype=np.float32)
        else:
            print(f"Input type unsupported: {inp.type}")
            exit()

    print(f"Running {model} {N} times")
    beg = time.time()
    for i in range(N):
        result = session.run(None, inputs)
    end = time.time()
    print(f"- Duration: {end - beg}")
    duration_map[model] = (end - beg) / N

max = max(duration_map.values())
duration_map = dict(sorted(duration_map.items()))
print(f"\n")
print(f"Performance as precent of maximum:")
for item in duration_map:
    print(f"{item} ::\
            \n- {duration_map[item] / max * 100.0 : 3.2f}% of maximum runtime\
            \n- {(max - duration_map[item]) / max * 100.0 : 3.2f}% faster than maximum\
            \n- {duration_map[item] : 2.5f} seconds")

