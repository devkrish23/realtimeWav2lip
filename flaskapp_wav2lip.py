from flask import Flask, Response, render_template, redirect, request, jsonify
import cv2
import os
from inference import main
import time

app=Flask(__name__) 

app.config['IMAGE_DIR'] = './assets/uploaded_images/' 
app.config['Filename'] = ''

@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the file upload
@app.route('/upload', methods=['POST'])
def upload():
    # Check if the POST request has the file part
    if 'image' not in request.files:
        return 'No file part'

    file = request.files['image']

    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return 'No selected file'

    # Save the file to the asset folder
    app.config['Filename'] = file.filename
    file.save(os.path.join(app.config['IMAGE_DIR'], file.filename))

    return redirect("/")

global flag
flag = 0

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global flag
    try:    
        if request.method == 'POST':
            if request.form.get('start') == 'Start':
                flag = 1
            elif  request.form.get('stop') == 'Stop':
                flag = 0
            elif  request.form.get('clear') == 'clear':
                flag = 0
            print(f"Flag value {flag}")
            time.sleep(2)
        elif request.method=='GET':
            return render_template('index.html')
    except Exception as e:
        print(e)

    return render_template("index.html")


@app.route('/video_feed', methods=['POST', 'GET'])
def video_feed():
    global flag
    try:    
        if app.config['Filename']!='':        
            return Response(main(os.path.join(app.config['IMAGE_DIR'], app.config['Filename']), flag) ,mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(e)
    return ""

if __name__=="__main__":
    app.run(debug=True)
