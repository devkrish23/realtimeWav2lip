# Use a base image with Python and necessary dependencies
FROM python:3.8-slim

LABEL maintainer="krishna158@live.com"


# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV FLASK_RUN_HOST 0.0.0.0
ENV ALSA_CONFIG_PATH=/etc/asound.conf

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt /app/requirements.txt

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        libportaudio2 \
        libportaudiocpp0 \
        portaudio19-dev \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        tk \
        alsa-utils \
        libasound2-dev\
        libsndfile1\
        wget

RUN apt-get install gcc -y

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask application into the container
COPY . /app

# Expose port 5000 to the outside world
EXPOSE 8080

# # Command to run the Flask application
# ENV FLASK_APP=flaskapp_wav2lip.py
# CMD ["flask", "run"]
ENTRYPOINT ["python"]
CMD ["flaskapp_wav2lip.py"]