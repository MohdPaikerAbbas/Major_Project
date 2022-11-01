from imutils.video import VideoStream
from flask import Response
import threading
import argparse
import imutils
import numpy as np
import time
import cv2

from flask import Flask, render_template, request, redirect, flash, session
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from scipy.spatial import distance
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# mediapipe face detection
def detect_face(frame, face_detection):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return results, frame

db = SQLAlchemy()

'''
to create the project database, open terminal
- type python and press enter
- type 
    from app import app, db
    with app.app_context():
        db.create_all()
- enter twice to confirm
'''

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(64), nullable=False)
    created_on = db.Column(db.DateTime, default=datetime.now)

    def __str__(self):
        return f'{self.username}({self.id})'

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database/app.sqlite'
    app.config['SQLALCHEMY_ECHO'] = True
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.secret_key = 'supersecretkeythatnooneknows'
    db.init_app(app)
    return app

app = create_app()

def create_login_session(user: User):
    session['id'] = user.id
    session['username'] = user.username
    session['email'] = user.email
    session['is_logged_in'] = True

def destroy_login_session():
    if 'is_logged_in' in session:
        session.clear()

def load_keras_model():
    print("LOG:", "Loading Keras model...")
    from keras.models import load_model
    path = 'models/masknet.h5'
    model = load_model(path)
    return model


# main part of the app

outputFrame = None
lock = threading.Lock()
vs = VideoStream(src=0).start()
time.sleep(2.0)

mask_label = {0:'MASK',1:'NO MASK'}
dist_label = {0:(0,255,0), 1:(255,0,0)}
MIN_DISTANCE = 130
model = load_keras_model()
mask_image_size = (128, 128)
font = cv2.FONT_HERSHEY_SIMPLEX



# detect mask and distance between faces
def detect_mask(image, detection):
    global model, mask_label, dist_label, MIN_DISTANCE
    image.flags.writeable = True
    bbox = detection.location_data.relative_bounding_box
    h, w, c = image.shape
    bbox = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
    try:
        face = image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        face = cv2.resize(face, mask_image_size)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = np.reshape(face, (1, 128, 128, 3))
        face = face / 255.0
        pred = model.predict(face)
        mask = mask_label[np.argmax(pred)]
        color = dist_label[np.argmax(pred)]
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color, 2)
        cv2.putText(image, mask, (bbox[0], bbox[1]-10), font, 0.5, color, 2)
        
        
    except:
        pass
    return image

def detect_faces(frameCount):
    global vs, outputFrame, lock
    total = 0
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while True:
            image = vs.read()
            timestamp = datetime.now()
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)
            if results.detections:
                boxes = []
                for detection in results.detections:
                    image = detect_mask(image, detection)
                    # detect distance between faces
                    bbox = detection.location_data.relative_bounding_box
                    h, w, c = image.shape
                    bbox = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                    boxes.append(bbox)
                for i in range(len(boxes)):
                    for j in range(i+1, len(boxes)):
                        d = distance.euclidean(boxes[i], boxes[j])

                        if d < MIN_DISTANCE:
                            cv2.rectangle(image, (boxes[i][0], boxes[i][1]), (boxes[i][0]+boxes[i][2], boxes[i][1]+boxes[i][3]), (0,0,255), 2)
                            cv2.rectangle(image, (boxes[j][0], boxes[j][1]), (boxes[j][0]+boxes[j][2], boxes[j][1]+boxes[j][3]), (0,0,255), 2)
                            cv2.putText(image, "WARNING", (boxes[i][0], boxes[i][1]-10), font, 0.5, (0,0,255), 2)
                            cv2.putText(image, "WARNING", (boxes[j][0], boxes[j][1]-10), font, 0.5, (0,0,255), 2)
                        else:
                            cv2.putText(image, "SAFE", (boxes[i][0], boxes[i][1]-10), font, 0.5, (0,255,0), 2)
                
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.putText(image, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"), (10, image.shape[0] - 10), font, 0.5, (255, 255, 255), 1)
            total += 1
            with lock:
                outputFrame = image.copy()

def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +  bytearray(encodedImage) + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/video')
def video_output():
    print(session.get('username','not available'))
    if not session.get('is_logged_in', False):
        return redirect('/login')
    return render_template('video.html')

# froute
@app.route('/login',  methods=['GET','POST'])
def login():
    errors = {}
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        print("LOGGIN IN",email, password)
        if password and email:
            if len(email) < 11 or '@' not in email:
                errors['email'] = 'Email is Invalid'
            if len(errors) == 0:
                user = User.query.filter_by(email=email).first()
                if user is not None:
                    print("user account found", user)
                    if user.password == password:
                        create_login_session(user)
                        flash('Login Successfull', "success")
                        return redirect('/')
                    else:
                        errors['password'] = 'Password is invalid'
                else:
                    errors['email']= 'Account does not exists'
        else:
            errors['email'] = 'Please fill valid details'
            errors['password'] = 'Please fill valid details'
    return render_template('login.html', errors = errors)

@app.route('/register', methods=['GET','POST'])
def register():
    errors = []
    if request.method == 'POST': # if form was submitted
        username = request.form.get('username')
        email = request.form.get('email')
        pwd = request.form.get('password')
        cpwd = request.form.get('confirmpass')
        print(username, email, pwd, cpwd)
        if username and email and pwd and cpwd:
            if len(username)<2:
                errors.append("Username is too small")
            if len(email) < 11 or '@' not in email:
                errors.append("Email is invalid")
            if len(pwd) < 6:
                errors.append("Password should be 6 or more chars")
            if pwd != cpwd:
                errors.append("passwords do not match")
            if len(errors) == 0:
                user = User(username=username, email=email, password=pwd)
                db.session.add(user)
                db.session.commit()
                flash('user account created','success')
                return redirect('/login')
        else:
            errors.append('Fill all the fields')
            flash('user account could not be created','warning')
    return render_template('register.html', error_list=errors)

@app.route('/logout')
def logout():
    destroy_login_session()
    flash('You are logged out','success')
    return redirect('/')    

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_faces, args=(32,))
    t.daemon = True
    t.start()
    # start the flask app
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True, use_reloader=False)
# release the video stream pointer
vs.stop()
# python app.py