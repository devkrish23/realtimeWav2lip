from flask import Flask, Response, render_template, redirect, request, jsonify
import os
from inference import main
import time

app = Flask(__name__)

app.config['MEDIA_DIR'] = './assets/uploaded_media/'
app.config['Filename'] = ''
app.config['RESULTS_DIR'] = './results/'

os.makedirs(app.config['RESULTS_DIR'], exist_ok=True)
os.makedirs(app.config['MEDIA_DIR'], exist_ok=True)

import sounddevice as sd

def get_audio_devices():
    try:
        devices = []
        for i, info in enumerate(sd.query_devices()):
            if info['max_input_channels'] > 0:
                devices.append({'id': i, 'name': info['name']})
        return devices
    except Exception as e:
        print(f"Error getting audio devices: {e}")
        return []

@app.route('/')
def index():
    devices = get_audio_devices()
    return render_template('index.html', devices=devices)

def remove_files_in_directory(directory):
    files = os.listdir(directory)
    for file in files:
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Route to handle the file upload
@app.route('/upload', methods=['POST'])
def upload():
    remove_files_in_directory(app.config['MEDIA_DIR'])

    if 'media' not in request.files:
        # Fallback to 'image' for backwards compatibility with older HTML
        if 'image' in request.files:
            file = request.files['image']
        else:
            return 'No file part'
    else:
        file = request.files['media']

    if file.filename == '':
        return 'No selected file'

    app.config['Filename'] = file.filename
    file.save(os.path.join(app.config['MEDIA_DIR'], file.filename))

    return redirect("/")

global flag
flag = 0
app.config['DEVICE_ID'] = None

@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global flag
    devices = get_audio_devices()
    try:
        if request.method == 'POST':
            if request.form.get('start') == 'Start':
                flag = 1
                device_id = request.form.get('audio_device')
                if device_id:
                    app.config['DEVICE_ID'] = int(device_id)
                noise_thresh = request.form.get('noise_threshold')
                if noise_thresh:
                    app.config['NOISE_THRESHOLD'] = int(noise_thresh)
                else:
                    app.config['NOISE_THRESHOLD'] = 400
            elif request.form.get('stop') == 'Stop':
                flag = 0
            elif request.form.get('clear') == 'clear':
                flag = 0
            print(f"Flag value {flag}, Device ID: {app.config.get('DEVICE_ID')}")
            time.sleep(2)
        elif request.method == 'GET':
            return render_template('index.html', devices=devices)
    except Exception as e:
        print(e)

    return render_template("index.html", devices=devices)


@app.route('/video_feed', methods=['POST', 'GET'])
def video_feed():
    if app.config['Filename'] != '':
        # Pass a callable so main() always reads the live flag value
        def get_flag():
            return flag

        results_dir = app.config['RESULTS_DIR']
        device_id = app.config.get('DEVICE_ID')
        noise_threshold = app.config.get('NOISE_THRESHOLD', 400)
        filepath = os.path.join(app.config['MEDIA_DIR'], app.config['Filename'])
        return Response(
            main(filepath, get_flag, results_dir, device_id, noise_threshold),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    return ""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
