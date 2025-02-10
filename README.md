# 2D Attention-based License Plate Recognition

The model with small backbone and 2D attention
Backbone: MobileNetV4-Small (modified) which is only 27.8 MFLOPs with 0.124 M parameters
Attention: 2D attention

| Model      | FLOPs         | Params    | Inference Time    | Accuracy  |
| :---:      | :---:         | :---:     | :---:             | :---:     |
| mbv4-tiny  | 16.64 MFLOPs  | 0.117 M   | 0.42 ms           | -%        |
| mbv4-small | 27.33 MFLOPs  | 0.172 M   | 0.46 ms           | 99.19%    |

Thus, the parameters are increased 47.01% and the FLOPs are increased 64.28% in new model.

| Model             | Chip          | Platform      | Params    | Inference Time    |
| :---:             | :---:         | :---:         | :---:     | :---:             |
| MobileNetV4-Small | i9-10900K     | onnx          | 0.172 M   | 0.46 ms           |
| MobileNetV4-Small | i9-10900K     | tflite        | 0.172 M   | 0.53 ms           |
| MobileNetV4-Small | M4            | tflite        | 0.172 M   | 0.58 ms           |
