import os, glob, sys, random
from time import perf_counter

import torch
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from PIL import Image


# onnx inference
def inference_onnx_model(model_path, img_path):
    img = Image.open(img_path).resize((96, 32)).convert('L')
    img = np.array(img).reshape(1, 1, 32, 96).astype(np.float32) / 255.0

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    # output_name = session.get_outputs()[0].name
    # Run the model
    output = session.run(None, {input_name: img})
    conf = np.max(output[0], axis=2).squeeze()
    # res = np.argmax(output[0], axis=2).squeeze()
    return output, conf


def test_onnx_model_speed(model_path, input_shape, warm_up=100, test=1000, force_cpu=True):
    # Set ONNX Runtime options for better performance
    options = ort.SessionOptions()
    options.intra_op_num_threads = os.cpu_count()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # ort devices check
    print(f'Available devices: {ort.get_device()}')

    # Create session with optimized options
    provider_options = ['CPUExecutionProvider'] if force_cpu else ort.get_available_providers()
    # check providers
    print(f'Available providers: {provider_options}')
    session = ort.InferenceSession(model_path, options, providers=provider_options)
    
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
    times = np.sort(times)[int(test * 0.2):int(test * 0.8)]
    average_time = np.mean(times)
    print(f'Average time: {average_time * 1000:.2f} ms')
    print(f'Min time: {min(times)*1000:.2f} ms')
    print(f'Max time: {max(times)*1000:.2f} ms')


if __name__ == '__main__':
    model_path = 'onnx/model_sim.onnx'
    model_path = 'model.onnx'
    # model_path = 'test_model.onnx'

    test_onnx_model_speed(model_path, (1, 1, 32, 96))

    img_path = random.choice(glob.glob('data/*.jpg'))
    img = Image.open(img_path).resize((96, 32)).convert('L')
    res, conf = inference_onnx_model(model_path, img_path)
    # print(img_path, res, conf)
    preds = np.argmax(res[0], axis=2).squeeze()

    # Prepare figure
    fig, axs = plt.subplots(1, 9, figsize=(16, 2))
    axs[0].imshow(img, cmap='gray')
    axs[0].axis('off')

    # Display attention maps
    for i, atten in enumerate(res[1:]):
        idx = i + 1
        atten = atten.squeeze()
        atten = (atten - atten.min()) / (atten.max() - atten.min())
        print("size: ", atten.shape)
        img = Image.fromarray((atten * 255).astype(np.uint8)).resize((96, 32), Image.NEAREST)
        axs[idx].set_title(preds[i])
        axs[idx].imshow(img, cmap='gray')
        axs[idx].axis('off')

    plt.tight_layout()
    plt.savefig('result.png')
    print(img_path, preds)
