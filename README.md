# Wav2Lip: Lip-Syncing Real-Time Audio with Images

## Introduction

Wav2Lip revolutionizes the realm of audio-visual synchronization with its groundbreaking real-time audio to video conversion capability. Powered by cutting-edge deep learning techniques, Wav2Lip accurately lip-syncs videos to any target speech in real-time, seamlessly aligning audio with visual content. This project leverages PyAudio for audio processing, Flask for server-side implementation, and a sophisticated inference mechanism that efficiently applies lip-syncing on images. Wav2Lip offers a user-friendly solution for generating lip movements from audio inputs, opening up possibilities for enhanced communication, entertainment, and creative expression.

| 📑 Original Paper | 🌀 Demo | 📔 Colab Notebook |
|:-----------------:|:------:|:-----------------:|
| [Paper](http://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/Projects/Speech-to-Lip/paper.pdf)        | [Demo](https://drive.google.com/file/d/1ACp7aDDOgchtABly4usLhmAAOGFpdq_c/view) | [Colab Notebook](https://colab.research.google.com/drive/15jHVLxYJvmptoYmlfpOGbNi0jSZ85hqq#scrollTo=sh72cJ0K-dfb) |

# Wav2Lip Installation and Usage Guide

## Installations
- Python 3.6
- Install necessary packages using `pip install -r requirements.txt`

## Python Libraries
- numpy: For numerical operations.
- opencv-python: For image processing and video I/O.
- pandas: For data manipulation.
- torch: PyTorch deep learning framework.
- tqdm: For progress bars.
- openvino: OpenVINO toolkit for optimized inference.
- pyaudio: For audio processing.
- Pillow: For image manipulation.

## Models and Files
- Wav2Lip directory containing the checkpoints directory with wav2lip_gan.pth.
- Pre-trained face detection model (mobilenet.pth) in the checkpoints directory.
- Pre-trained OpenVINO model (wav2lip_openvino_model.xml) in the openvino_model directory.
- An image of the face to sync with the audio (Elon_Musk.jpg).

## Optional Dependencies
- platform: For platform-specific operations (optional).
- subprocess: For subprocess management (optional).


# Model Weights

| Model                     | Description                                                     | Link to the model                                        |
|---------------------------|-----------------------------------------------------------------|----------------------------------------------------------|
| Wav2Lip                   | Highly accurate lip-sync                                        | [Download](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?e=TBFBVW)                                         |
| Wav2Lip + GAN             | Slightly inferior lip-sync, but better visual quality           | [Download](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW)                                         |
| Expert Discriminator      | Weights of the expert discriminator                             | [Download](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EQRvmiZg-HRAjvI6zqN9eTEBP74KefynCwPWVmF57l-AYA?e=ZRPHKP)                                         |
| Visual Quality Discriminator | Weights of the visual disc trained in a GAN setup             | [Download](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EQVqH88dTm1HjlK11eNba5gBbn15WMS0B0EZbDBttqrqkg?e=ic0ljo)                                         |
| Face Detection Model | Face detection model weights | [Download](https://drive.google.com/drive/u/0/folders/1BopYvKEVgPK23t3rAR1kBge77N9NlP7p)                 |
| Wav2Lip + GAN (OpenVino) | Inferior lip-sync, but better real-time performance | [Download](https://drive.google.com/drive/folders/193qN6CXkuDorYOHuVj-qDQmI0MLDHlu-)




## Real-time Audio Capture and Processing

In real-time lip-syncing inference, we capture audio using PyAudio while simultaneously processing image frames. PyAudio streams audio data in chunks, with each chunk representing a small piece of the audio input. As the audio stream is captured, we divide it into mel-spectrogram chunks, which represent the frequency content of the audio over time. These mel-spectrogram chunks are then fed into the lip-syncing model along with corresponding image frames. The lip-syncing model generates lip movements synchronized with the audio, which are then overlaid onto the image frames. This process continues iteratively for each audio chunk, allowing for real-time lip-syncing of the image based on the captured audio input.

## Requirements
- Python 3.x
- PyAudio
- Flask

## Code
```python
import pyaudio
import numpy as np
from time import time

CHUNK = 1024  # Number of frames per buffer during audio capture
FORMAT = pyaudio.paInt16  # Format of the audio stream
CHANNELS = 1  # Number of audio channels (1 for monaural audio)
RATE = 16000  # Sample rate of the audio stream (16000 samples/second)
RECORD_SECONDS = 0.5  # Duration of audio capture in seconds

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

def record_audio_stream():
    frames = []
    print("Recording audio ...")
    start_time = time()
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    end_time = time()
    print("Recording time:", end_time - start_time, "seconds")

    # Combine all recorded frames into a single numpy array
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    return audio_data

# Example usage
audio_data = record_audio_stream()
# Now you can process the audio data as needed, such as generating mel-spectrogram chunks for lip-syncing inference
```

# Optimizaton 

We are using OpenVINO to optimize the inference process of the Wav2Lip model for lip-syncing videos. OpenVINO, or Open Visual Inference and Neural network Optimization, provides a set of tools and libraries to accelerate deep learning inference across a variety of Intel hardware, including CPUs, GPUs, and VPUs. By leveraging OpenVINO, we can optimize the model for inference on Intel processors, improving performance and reducing latency. This is particularly useful for real-time applications like lip-syncing, where low latency is crucial for a seamless user experience.

# Model Conversion using OpenVINO

To convert a PyTorch model to an OpenVINO model using ONNX as an intermediate representation, follow these steps:

1. Conversion of Pytorch to ONNX:
    - Load the PyTorch model.
    - Load the trained weights into the model.
    - Prepare sample input data.
    - Export the model to ONNX format using torch.onnx.export().
      
2. Conversion of ONNX to OpenVINO:
    - Import OpenVINO library.
    - Read the ONNX model.
    - Compile the model for a specific target device using core.compile_model() function.
   
##Code
```python
import torch
from models import Wav2Lip
import openvino as ov
import os
import numpy as np


device = 'cpu'

onnx_model_path = 'Wav2Lip/openvino_model/wav2lip_onnx_export.onnx'

# Conversion of Pytorch to ONNX
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

# Conversion of ONNX to OpenVINO
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

convert_pytorch_to_onnx(onnx_model_path)
convert_onnx_to_openvino()
print("successfully converted pytorch -> onnx -> openvino")
```

Reference: Detailed instructions for model conversion can be found here - [link](https://docs.openvino.ai/2024/get-started.html)
# Real-time Lip-Syncing App with Flask

To run this Flask app, first ensure you have Python installed along with the necessary dependencies such as Flask and PyAudio. Then follow these steps:

1. **Clone the Repository**: Clone the repository containing the Flask app and navigate to its directory.

2. **Create virtual environments for python with conda**:
    - Open Conda terminal client and check if Conda is available by running `conda -V`.
    - Run `conda update conda` to check Conda is up to date.
    - Create virtual environment by running `conda create -n <your env name>`.
    - Run `conda activate <your env name>` to activate your virtual enviornment.
    - Run `conda install pip`. This will install pip to your virtual environment directory.
    - Navigate to the directory containing cloned repository using `cd <path to directory>`

3. **Install Dependencies**: Install the required Python dependencies by running `pip install -r requirements.txt`.

4. **Run the App**: Execute the `app.py` file to start the Flask app. You can do this by running `python app.py` in your terminal.

5. **Access the Web Interface**: Open a web browser and navigate to `http://localhost:8080` to access the app's interface.

6. **Upload an Image**: Use the interface to upload an image containing a face.

7. **Start Lip-Syncing**: Click on the "Start" button to start the lip-syncing process. The lip-synced video will appear on the same page.

8. **Stop Lip-Syncing**: Click on the "Stop" button to pause the lip-syncing process.

9. **Clear Image**: To upload a new image, click on the "Clear" button to remove the current image.

By following these steps, you can use Flask to run the app and perform real-time lip-syncing inference based on the uploaded image.

## Real-Time Lip-Syncing Flask App

This Flask application offers a real-time lip-syncing solution, allowing users to upload an image and generate a lip-synced video. The application consists of three main routes:

- **/upload**: This route handles file uploads. When a user uploads an image file, it clears any existing images in the specified directory, saves the uploaded file to that directory, and sets the `Filename` configuration variable to the filename of the uploaded file. Afterward, it redirects the user to the root URL.

- **/requests**: This route manages the lip-syncing process. It handles both GET and POST requests for controlling the lip-syncing operation. When a POST request is made, it checks the form data to start, stop, or clear the lip-syncing process by setting the `flag` variable accordingly. A sleep is included to prevent rapid flag changes. For GET requests, it renders the index template, which contains controls for starting, stopping, and clearing the lip-syncing process.

- **/video_feed**: This route streams the lip-synced video. If an image has been uploaded, it returns a response containing the output of the `main` function, which generates the lip-synced video. The response is of type `multipart/x-mixed-replace`, enabling continuous streaming of video frames to the client.

```python
from flask import Flask, Response, render_template, redirect, request, jsonify
import os
from inference import main
import time

app = Flask(__name__) 

app.config['IMAGE_DIR'] = './assets/uploaded_images/' 
app.config['Filename'] = ''

# Function to remove files in a directory
def remove_files_in_directory(directory):
    files = os.listdir(directory)
    for file in files:
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    remove_files_in_directory(app.config['IMAGE_DIR'])

    if 'image' not in request.files:
        return 'No file part'

    file = request.files['image']

    if file.filename == '':
        return 'No selected file'

    app.config['Filename'] = file.filename
    file.save(os.path.join(app.config['IMAGE_DIR'], file.filename))

    return redirect("/")

global flag
flag = 0

@app.route('/requests', methods=['POST','GET'])
def tasks():
    global flag
    try:    
        if request.method == 'POST':
            if request.form.get('start') == 'Start':
                flag = 1
            elif request.form.get('stop') == 'Stop':
                flag = 0
            elif request.form.get('clear') == 'clear':
                flag = 0
            print(f"Flag value {flag}")
            time.sleep(2)
        elif request.method == 'GET':
            return render_template('index.html')
    except Exception as e:
        print(e)

    return render_template("index.html")


@app.route('/video_feed', methods=['POST', 'GET'])
def video_feed():
    global flag
    try:    
        if app.config['Filename'] != '':        
            return Response(main(os.path.join(app.config['IMAGE_DIR'], app.config['Filename']), flag), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(e)
    return ""
```

## Benchmark
|    Model    |    Inference Time    |
|-------------|----------------------|
| Wav2Lip + GAN    |    0.3 sec    |
| Wav2Lip + GAN (OpenVino) |    0.26 sec    |

Optimized the code using OpenVino model conversion. 
The wav2lip+GAN (OpenVino) model does not require GPU and has less inference time as mentioned in the above Benchmark table.

