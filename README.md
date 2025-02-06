# 2D Attention-based License Plate Recognition

The model with small backbone and 2D attention
Backbone: MobileNetV4-Small (modified) which is only 27.8 MFLOPs with 0.124 M parameters
Attention: 2D attention

| Backbone          | Attention     | FLOPs         | Params    | Inference Time    | Accuracy  |
| :---:             | :---:         | :---:         | :---:     | :---:             | :---:     |
| MobileNetV4-Small | -             | 25.47 MFLOPs  | 0.110 M   | 0.83 ms           | -%        |
| MobileNetV4-Small | 2D attention  | 27.33 MFLOPs  | 0.172 M   | 0.848 ms          | 99.19%    |

Thus, the parameters are increased 56.36% and the flops are increased 7.27%.

| Model             | Chip          | Platform      | Params    | Inference Time    |
| :---:             | :---:         | :---:         | :---:     | :---:             |
| MobileNetV4-Small | i9-10900K     | onnx          | 0.172 M   | 0.46 ms           |
| MobileNetV4-Small | M4            | tflite        | 0.172 M   | 0.58 ms           |
