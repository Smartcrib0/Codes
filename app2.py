from flask import Flask, request, jsonify
import numpy as np
import cv2
from ultralytics import YOLO
import os
import librosa
import sounddevice as sd
import threading
import wavio
import base64
from tensorflow.keras.models import load_model
import uuid
import time
import firebase_admin
from firebase_admin import credentials, db,storage

# تهيئة تطبيق Flask
app_video = Flask(__name__)
app_audio = Flask(__name__)
app_sensor = Flask(__name__)

# تحميل نماذج YOLO وصوت البكاء وسبب البكاء
model_path_audio = 'E:/Senior_Project/audio_classifier_mfcc_improved.h5'
model_audio = load_model(model_path_audio)
CATEGORIES = ['Crying', 'Laugh', 'Noise', 'Silence']

# تحميل النموذج المدرب الثاني لتحديد سبب البكاء
model_path_cry_reason = 'E:/Senior_Project/sound_detection_modelCNN.h5'
model_cry_reason = load_model(model_path_cry_reason)
classes = ['belly_pain', 'burping', 'cold-hot', 'discomfort', "dontKnow", 'hungry', 'lonely', 'scared', 'tired']

# مسار نموذج YOLO
model_yolo = YOLO("yolo11n.pt")

# تحميل إعدادات Firebase Admin SDK
cred = credentials.Certificate("E:/Senior_Project/RPi/smart-baby-nest-firebase-adminsdk.json")  # استبدل هذا بمسار ملف خدمة Firebase
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://smart-baby-nest-default-rtdb.asia-southeast1.firebasedatabase.app/'  # URL الخاص بقاعدة البيانات
})

 
# مرجع Firebase للإطارات
# frames_ref = db.reference('/live_stream/frames')

class AudioProcessor():
    cry_prediction = ""
    cry_reason = ""
    is_processing = False

    def process_audio_file(self, filename):
        try:
            print("Processing audio file...")
            y, sr = librosa.load(filename, sr=None)
            energy = np.sum(y**2) / len(y)  # الطاقة
            rms = librosa.feature.rms(y=y)[0].mean()  # الجهارة
            zcr = librosa.feature.zero_crossing_rate(y)[0].mean()  # معدل عبور الصفر

            # التحقق من السكون بناءً على الطاقة والجهارة
            if energy < 1e-5 and rms < 0.01:
                self.cry_prediction = "Silence"
                self.cry_reason = "No Reason"
            elif zcr < 0.02 and rms < 0.015:
                self.cry_prediction = "Silence"
                self.cry_reason = "No Reason"
            else:
                cry_prediction = predict_cry(filename)
                self.cry_prediction = cry_prediction
                if cry_prediction == 'Crying':
                    cry_reason = predict_cry_reason(filename)
                    self.cry_reason = cry_reason
                else:
                    self.cry_prediction = "Other Sound"
                    self.cry_reason = ""
        except Exception as e:
            print(f"Error in audio processing: {e}")
        finally:
            print("Audio is self.processing .")
            self.is_processing = False

def predect_audio(filepath):
    audio_processor = AudioProcessor()
    audio_processor.process_audio_file(filepath)
    frame_prediction_cry = audio_processor.cry_prediction
    frame_prediction_reason = audio_processor.cry_reason
    print("frame_prediction_cry", frame_prediction_cry)
    print("frame_prediction_reason", frame_prediction_reason)
    return frame_prediction_cry, frame_prediction_reason


@app_sensor.route('/upload_sensor_data', methods=['POST'])
def upload_sensor_data():
    data = request.json
    temperature = data.get('temperature')
    humidity = data.get('humidity')

    # التحقق من أن جميع البيانات موجودة
    if None in (temperature, humidity):
        return jsonify({"error": "Missing data"}), 400
  
    #DC Motor: {dc_motor}, Servo Motor: {servo_motor}, 
    print(f"Temperature: {temperature} °C, Humidity: {humidity} %")

    # تخزين البيانات في Firebase Realtime Database
    ref = db.reference('/sensor_data')  # المسار في قاعدة البيانات حيث سيتم تخزين البيانات
    sensor_data = {
        'temperature': temperature,
        'humidity': humidity,
        'timestamp': time.time()  # إضافة التوقيت لكل قياس
    }
    ref.push(sensor_data)  # إضافة البيانات الجديدة إلى قاعدة البيانات
    
    return jsonify({"status": "success"}), 200


@app_video.route('/upload_frame', methods=['POST'])
def upload_frame():
    file = request.files['frame']
    np_img = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    child_detected = detect_child(frame)
    
    # _, buffer = cv2.imencode('.jpg', frame)
    # base64_image = base64.b64encode(buffer).decode('utf-8')

    # # تخزين الفريم في Firebase Realtime Database
    # ref = db.reference('/frames')  # المسار في قاعدة البيانات
    # frame_data = {
    #     'image': base64_image,
    #     'timestamp': time.time()  # إضافة التوقيت لكل فريم
    # }
    # ref.push(frame_data)  # إضافة البيانات الجديدة إلى قاعدة البيانات

    # print("Frame uploaded to Firebase Realtime Database")

    # # كشف وجود الطفل
    # child_detected = detect_child(frame)
    
    return jsonify({"child_detected": child_detected})

def detect_child(frame):
    results = model_yolo(frame)
    for box in results[0].boxes:
        cls = box.cls
        conf = box.conf
        if int(cls[0]) == 67 and conf[0] > 0.70:
            return True
    return False

@app_audio.route('/analyze_audio', methods=['POST'])
def upload_audio():
    # التحقق من وجود الملف في الطلب
    if 'audio' not in request.files:
        print("Error: No file part in the request.")
        return "No file part", 400

    file = request.files['audio']  # الحصول على الملف من الطلب
    if file.filename == '':
        print("Error: No selected file.")
        return "No selected file", 400

    print("Received audio file from Raspberry")
    audio_filename = "received_audio.wav"

    # حفظ الملف الصوتي
    try:
        file.save(audio_filename)
        print(f"Saved audio file to {audio_filename}")
    except Exception as e:
        print(f"Error saving file: {e}")
        return "Error saving file", 500

    # تحليل الصوت
    try:
        result = analyze_audio(audio_filename)
        print(f"Analysis result sent to client: {result.json}")
    except Exception as e:
        print(f"Error during analysis: {e}")
        return "Error during analysis", 500
    finally:
        # حذف الملف الصوتي بعد المعالجة
        if os.path.exists(audio_filename):
            print("the file is exist")

    return result

# تحليل الصوت (وظيفة مساعدة)
def analyze_audio(audio_filename):
    print("Processing audio file...")
    cry_prediction, cry_reason = predect_audio(audio_filename)

    # طباعة النتائج بشكل واضح
    print(f"Prediction result: {cry_prediction}")
    if not cry_reason:
        print("Warning: No specific reason detected from audio analysis.")
        cry_reason = "Unknown reason"

    return jsonify({"Other Sound": "yes" if cry_prediction == "Other Sound" else "no", "reason": cry_reason})


def predict_cry(file_path):
    mfcc = process_audio_mfcc(file_path)
    mfcc = np.expand_dims(mfcc, axis=0)
    prediction = model_audio.predict(mfcc)
    return CATEGORIES[np.argmax(prediction)]


def process_audio_mfcc(file_path, duration_seconds=10, target_sr=22050):
    y, sr = librosa.load(file_path, sr=target_sr)
    y = librosa.util.fix_length(y, size=target_sr * duration_seconds)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc = np.mean(mfcc.T, axis=0)
    return np.expand_dims(mfcc, axis=-1)


def predict_cry_reason(file_path):
    mel_spec = process_audio_mel(file_path)
    mel_spec = np.expand_dims(mel_spec, axis=-1)
    mel_spec = np.expand_dims(mel_spec, axis=0)
    mel_spec = mel_spec / np.max(mel_spec)
    prediction = model_cry_reason.predict(mel_spec)
    return classes[np.argmax(prediction)]


def process_audio_mel(file_path, duration_seconds=6, target_sr=22050):
    y, sr = librosa.load(file_path, sr=target_sr)
    y = librosa.util.fix_length(y, size=target_sr * duration_seconds)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    return librosa.power_to_db(mel_spec, ref=np.max)


def run_app(app, port):
    app.run(host='0.0.0.0', port=port)
    
if __name__ == '__main__':
    video_thread = threading.Thread(target=run_app, args=(app_video, 5000), daemon=True)
    audio_thread = threading.Thread(target=run_app, args=(app_audio, 5001), daemon=True)
    sensor_thread = threading.Thread(target=run_app, args=(app_sensor, 5050), daemon=True)
    
    video_thread.start()
    audio_thread.start()
    sensor_thread.start()

    video_thread.join()
    audio_thread.join()
    sensor_thread.join()
    
    try:
        while True:
            time.sleep(1)  # ضمان استمرار البرنامج
    except KeyboardInterrupt:
        print("Shutting down...")