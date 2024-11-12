import torch
from thop import profile
# import torch.onnx as onnx
import onnx
from onnxsim import simplify


def count_parameters(model, input_size=(1, 3, 224, 224)):
    x = torch.randn(input_size)
    macs, params = profile(model, inputs=(x,), verbose=False)
    print('MACs: {} MFLOPs'.format(macs / 1e6))
    print('Params: {} M'.format(params / 1e6))


def export2onnx(model, input_size=(1, 3, 224, 224), model_name="mobilenetv4_small.onnx"):
    model.eval()
    x = torch.randn(input_size)
    torch.onnx.export(model, x,
        model_name,
        verbose=False,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
    )


def simplify_onnx(original_model, simplified_model):
    model = onnx.load(original_model)
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, simplified_model)
