import numpy as np
import tensorflow as tf
import time


def load_and_test_tflite_model(tflite_model_path, input_shape, warmup=100, num_iterations=1000):
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Create random input data
    input_data = np.random.random(input_shape).astype(np.float32)

    # Warm-up run
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    # Warmup runs
    for _ in range(warmup):
        input_data = np.random.random(input_shape).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])

    # Test inference time
    times = []
    for i in range(num_iterations):
        # Generate new random input for each iteration
        input_data = np.random.random(input_shape).astype(np.float32)
        
        start_time = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
        end_time = time.perf_counter()
        
        times.append((end_time - start_time) * 1000)  # Store time in milliseconds

    # Sort times and remove top and bottom 10%
    times.sort()
    cut_size = int(num_iterations * 0.1)
    filtered_times = times[cut_size:-cut_size]
    
    # print max and min and avg time
    max_time = max(filtered_times)
    min_time = min(filtered_times)
    avg_time = sum(filtered_times) / len(filtered_times)
    print(f"Max time: {max_time:.2f} ms")
    print(f"Min time: {min_time:.2f} ms")
    print(f"Average time: {avg_time:.2f} ms")
    return avg_time


if __name__ == "__main__":
    # Example usage
    model_path = "models/model.tflite"
    input_shape = (1, 1, 32, 96)  # Modify according to your model's input shape

    avg_inference_time = load_and_test_tflite_model(model_path, input_shape)
    print(f"Average inference time: {avg_inference_time:.2f} ms")
