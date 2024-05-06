# Wav2Lip: Lip-Syncing real-Time Audio with Images

## Introduction

Wave2Lip revolutionizes the realm of audio-visual synchronization with its groundbreaking real-time audio to video conversion capability. Powered by cutting-edge deep learning techniques, Wave2Lip accurately lip-syncs videos to any target speech in real-time, seamlessly aligning audio with visual content. This project leverages PyAudio for audio processing, Flask for server-side implementation, and a sophisticated inference mechanism that efficiently applies lip-syncing on images. Wave2Lip offers a user-friendly solution for generating lip movements from audio inputs, opening up possibilities for enhanced communication, entertainment, and creative expression.

| 📑 Original Paper | 🌀 Demo | 📔 Colab Notebook |
|:-----------------:|:------:|:-----------------:|
| [Paper](http://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/Projects/Speech-to-Lip/paper.pdf)        | [Demo](https://drive.google.com/file/d/1ACp7aDDOgchtABly4usLhmAAOGFpdq_c/view) | [Colab Notebook](https://colab.research.google.com/drive/15jHVLxYJvmptoYmlfpOGbNi0jSZ85hqq#scrollTo=sh72cJ0K-dfb) |

# Wav2Lip Installation and Usage Guide

## Installations
- Python 3.6
- ffmpeg: `sudo apt-get install ffmpeg`
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
- tkinter: For GUI applications (optional).
- platform: For platform-specific operations (optional).
- subprocess: For subprocess management (optional).

## Usage
1. Ensure Python 3.6 is installed.
2. Install ffmpeg and other necessary packages.
3. Clone this repository and navigate to the root directory.
4. Install required Python libraries using `pip install -r requirements.txt`.
5. Place your audio file and image of the face in the appropriate directories.
6. Run the Wav2Lip program, providing the necessary arguments.

# Model Weights

| Model                     | Description                                                     | Link to the model                                        |
|---------------------------|-----------------------------------------------------------------|----------------------------------------------------------|
| Wav2Lip                   | Highly accurate lip-sync                                        | [Download](https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels%2Fwav2lip%2Epth&parent=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels&ga=1)                                         |
| Wav2Lip + GAN             | Slightly inferior lip-sync, but better visual quality           | [Download](https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels%2Fwav2lip%5Fgan%2Epth&parent=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels&ga=1)                                         |
| Expert Discriminator      | Weights of the expert discriminator                             | [Download](https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels%2Flipsync%5Fexpert%2Epth&parent=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels&ga=1)                                         |
| Visual Quality Discriminator | Weights of the visual disc trained in a GAN setup             | [Download](https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels%2Fvisual%5Fquality%5Fdisc%2Epth&parent=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels&ga=1)                                         |




## Real-time Audio Capture and Processing

In real-time lip-syncing inference, we capture audio using PyAudio while simultaneously processing image frames. PyAudio streams audio data in chunks, with each chunk representing a small piece of the audio input. As the audio stream is captured, we divide it into mel-spectrogram chunks, which represent the frequency content of the audio over time. These mel-spectrogram chunks are then fed into the lip-syncing model along with corresponding image frames. The lip-syncing model generates lip movements synchronized with the audio, which are then overlaid onto the image frames. This process continues iteratively for each audio chunk, allowing for real-time lip-syncing of the image based on the captured audio input.

## Requirements
- Python 3.x
- PyAudio

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



# Real-time Lip-Syncing App with Flask

To run this Flask app, first ensure you have Python installed along with the necessary dependencies such as Flask and PyAudio. Then follow these steps:

1. **Clone the Repository**: Clone the repository containing the Flask app and navigate to its directory.

2. **Install Dependencies**: Install the required Python dependencies by running `pip install -r requirements.txt`.

3. **Run the App**: Execute the `app.py` file to start the Flask app. You can do this by running `python app.py` in your terminal.

4. **Access the Web Interface**: Open a web browser and navigate to `http://localhost:8080` to access the app's interface.

5. **Upload an Image**: Use the interface to upload an image containing a face.

6. **Start Lip-Syncing**: Click on the "Start" button to start the lip-syncing process. The lip-synced video will appear on the same page.

7. **Stop Lip-Syncing**: Click on the "Stop" button to pause the lip-syncing process.

8. **Clear Image**: To upload a new image, click on the "Clear" button to remove the current image.

Here's an example usage scenario:

1. Upload an image by clicking the "Choose File" button.
2. Click "Start" to begin the lip-syncing process. The lip-synced video will start playing.
3. Click "Stop" to pause the lip-syncing process.
4. If you want to upload a different image, click "Clear" to remove the current image and upload a new one.

By following these steps, you can use Flask to run the app and perform real-time lip-syncing inference based on the uploaded image.


