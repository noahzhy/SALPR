# 2D Attention-based License Plate Recognition

The model with small backbone and 2D attention
Backbone: MobileNetV4-Small (modified) which is only 27.8 MFLOPs with 0.124 M parameters
Attention: 2D attention

| Model      | FLOPs    | Params  | Inference Time | Accuracy  |
| :---:      | :---:    | :---:   | :---:          | :---:     |
| mbv4-small | 54.66 M  | 172 K   | 0.46 ms        | 99.19%    |

## Speed Test

| Model             | Chip       | Platform  | Params  | Inference Time  |
| :---:             | :---:      | :---:     | :---:   | :---:           |
| MobileNetV4-Small | i9-10900K  | onnx      | 172 K   | 0.46 ms         |
| MobileNetV4-Small | M4         | onnx      | 172 K   | - ms            |
