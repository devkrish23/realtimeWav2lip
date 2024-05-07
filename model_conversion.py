import torch
from models import Wav2Lip
import openvino as ov
import os
import numpy as np


device = 'cpu'

onnx_model_path = 'Wav2Lip/openvino_model/wav2lip_onnx_export.onnx'

def convert_pytorch_to_onnx(onnx_model_path):
        
    torch_model = Wav2Lip()
    checkpoint = torch.load("Wav2Lip/checkpoints/wav2lip_gan.pth",
                            map_location=lambda storage, loc: storage)

    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    torch_model.load_state_dict(new_s)

    torch_model = torch_model.to("cpu")

    torch_model.eval()

    img_batch, mel_batch = np.random.rand(128, 6, 96, 96), np.random.rand(128, 1, 80, 16)
    img_batch = torch.FloatTensor(img_batch).to(device)
    mel_batch = torch.FloatTensor(mel_batch).to(device)
    print(img_batch.shape, mel_batch.shape)

    torch.onnx.export(torch_model,
                      (mel_batch, img_batch), 
                      onnx_model_path,
                      input_names = ["audio_sequences", "face_sequences"], 
                      output_names = ["output"],
                      dynamic_axes = {"audio_sequences": {0: "batch_size", 1: "time_size"}, "face_sequences": {0: "batch_size", 1: "channel"}},
                      )

def convert_onnx_to_openvino():

    core = ov.Core()

    devices = core.available_devices
    print(devices[0])

    for device in devices:
        device_name = core.get_property(device, "FULL_DEVICE_NAME")
        print(f"{device}: {device_name}")

    model_onnx = core.read_model(model=onnx_model_path)
    compiled_model_onnx = core.compile_model(model=model_onnx, device_name=devices[0])

    ov.save_model(model_onnx, output_model="wav2lip_openvino_model.xml")


def test_openvino_model():
    
    core = ov.Core()
    devices = core.available_devices
    
    print(devices[0])
    model = core.read_model(model="wav2lip_openvino_model.xml")

    img_batch, mel_batch = np.random.rand(128, 6, 96, 96), np.random.rand(128, 1, 80, 16)
    img_batch = torch.FloatTensor(img_batch).to("cpu")
    mel_batch = torch.FloatTensor(mel_batch).to("cpu")
    print(img_batch.shape, mel_batch.shape)
    print(model.inputs)

    compiled_model = core.compile_model(model = model, device_name = devices[0])
    result = compiled_model([mel_batch, img_batch])['output']
    print(result)

convert_pytorch_to_onnx(onnx_model_path)
convert_onnx_to_openvino()
print("successfully converted pytorch -> onnx -> openvino")
test_openvino_model()