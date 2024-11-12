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
    return np.argmax(output[0], axis=2)


# Load the ONNX model
def test_onnx_model_speed(model_path, input_shape, warm_up=100, test=1000):
    session = ort.InferenceSession(model_path)

    # Generate random input data
    input_name = session.get_inputs()[0].name
    input_data = np.random.random(input_shape).astype(np.float32)
    output_name = session.get_outputs()[0].name

    # Warm up and measure the time for 1000 iterations
    start_time = 0
    for i in range(warm_up + test):
        if i == warm_up - 1:
            start_time = perf_counter()
        session.run([output_name], {input_name: input_data})
    end_time = perf_counter()

    # Calculate and print the average time (ms) for each iteration
    average_time = (end_time - start_time) * 1000 / test
    print(f'Average time: {average_time} ms')


if __name__ == '__main__':
    test_onnx_model_speed('model.onnx', (1, 1, 32, 96))

    img_path = random.choice(glob.glob('images/*.jpg'))
    res = inference_onnx_model('model.onnx', img_path)
    print(img_path, res)
