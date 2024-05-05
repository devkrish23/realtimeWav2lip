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
| Wav2Lip                   | Highly accurate lip-sync                                        | [Download](link)                                         |
| Wav2Lip + GAN             | Slightly inferior lip-sync, but better visual quality           | [Download](link)                                         |
| Expert Discriminator      | Weights of the expert discriminator                             | [Download](link)                                         |
| Visual Quality Discriminator | Weights of the visual disc trained in a GAN setup             | [Download](link)                                         |
