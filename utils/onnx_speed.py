import os, glob, sys, random
from time import perf_counter

import numpy as np
import onnxruntime as ort
from PIL import Image


# onnx inference
def inference_onnx_model(model_path, img_path):
    img = Image.open(img_path).resize((96, 32)).convert('L')
    img = np.array(img).reshape(1, 1, 32, 96).astype(np.float32) / 255.0

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    # Run the model
    output = session.run([output_name], {input_name: img})
    conf = np.max(output[0], axis=2).squeeze()
    res = np.argmax(output[0], axis=2).squeeze()
    return res, conf


def test_onnx_model_speed(model_path, input_shape, warm_up=100, test=1000):
    # Set ONNX Runtime options for better performance
    options = ort.SessionOptions()
    options.intra_op_num_threads = os.cpu_count()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Create session with optimized options
    session = ort.InferenceSession(model_path, options)
    
    # Pre-allocate input data array
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_data = np.random.random(input_shape).astype(np.float32)

    # Warm up phase
    for _ in range(warm_up):
        session.run([output_name], {input_name: input_data})

    # Test phase with timing
    times = []
    for _ in range(test):
        start = perf_counter()
        session.run([output_name], {input_name: input_data})
        times.append(perf_counter() - start)

    # Calculate statistics
    # sort the times and remove the first 10% and last 10% of the times
    times = np.sort(times)[int(test * 0.1):int(test * 0.9)]
    average_time = np.mean(times)
    print(f'Average time: {average_time * 1000:.2f} ms')
    print(f'Min time: {min(times)*1000:.2f} ms')
    print(f'Max time: {max(times)*1000:.2f} ms')


if __name__ == '__main__':
    model_path = 'model.onnx'
    # model_path = 'onnx/model.onnx'

    test_onnx_model_speed(model_path, (1, 1, 32, 96))

    img_path = random.choice(glob.glob('data/*.jpg'))
    res, conf = inference_onnx_model(model_path, img_path)
    print(img_path, res, conf)
