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
| Wav2Lip                   | Highly accurate lip-sync                                        | [Download](https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels%2Fwav2lip%2Epth&parent=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels&p=14)                                         |
| Wav2Lip + GAN             | Slightly inferior lip-sync, but better visual quality           | [Download](https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels%2Fwav2lip%5Fgan%2Epth&parent=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels&ga=1)                                         |
| Expert Discriminator      | Weights of the expert discriminator                             | [Download](https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels%2Flipsync%5Fexpert%2Epth&parent=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels&ga=1)                                         |
| Visual Quality Discriminator | Weights of the visual disc trained in a GAN setup             | [Download](https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels%2Fvisual%5Fquality%5Fdisc%2Epth&parent=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels&ga=1)                                         |



## Real-time Audio Capture and Processing

In real-time lip-syncing inference, we capture audio using PyAudio while simultaneously processing video frames. PyAudio streams audio data in chunks, with each chunk representing a small piece of the audio input. As the audio stream is captured, we divide it into mel-spectrogram chunks, which represent the frequency content of the audio over time. These mel-spectrogram chunks are then fed into the lip-syncing model along with corresponding video frames. The lip-syncing model generates lip movements synchronized with the audio, which are then overlaid onto the video frames. This process continues iteratively for each audio chunk, allowing for real-time lip-syncing of the video based on the captured audio input

## Requirements
- Python 3.x
- PyAudio

## Usage
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



