from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import datetime
import pickle
import cv2
import numpy as np
import base64
import face_recognition
from SilentFaceAntiSpoofing.test import test
import util

app = Flask(__name__)
CORS(app)

db_dir = './db'
if not os.path.exists(db_dir):
    os.mkdir(db_dir)

log_path = './log.txt'

def process_frame(frame):
    label = test(
        image=frame,
        model_dir='SilentFaceAntiSpoofing/resources/anti_spoof_models',
        device_id=0
    )
    if label == 1:
        name = util.recognize(frame, db_dir)
        if name in ['unknown_person', 'no_persons_found']:
            return {'status': 'error', 'message': 'Unknown user. Please register new user or try again.'}
        else:
            with open(log_path, 'a') as f:
                f.write('{},{},in\n'.format(name, datetime.datetime.now()))
            return {'status': 'success', 'message': f'Welcome, {name}.', 'name': name}
    else:
        return {'status': 'error', 'message': 'You are a spoofer!'}

@app.route('/login', methods=['POST'])
def login():
    print("here")
    data = request.get_json()
    base64_image = data['image']
    image_data = base64.b64decode(base64_image)
    np_arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    result = process_frame(frame)
    return jsonify(result)

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    base64_image = data['image']
    username = data['username']
    image_data = base64.b64decode(base64_image)
    np_arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    embeddings = face_recognition.face_encodings(frame)[0]
    with open(os.path.join(db_dir, f'{username}.pickle'), 'wb') as f:
        pickle.dump(embeddings, f)
    return jsonify({'status': 'success', 'message': 'User registered successfully!'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
