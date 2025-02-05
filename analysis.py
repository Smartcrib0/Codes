import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# مسار البيانات الصوتية والفئات
DATA_PATH = 'E:/Senior_Project/Sound_Detection/soundAnalyzer/data'
CATEGORIES = ['Crying', 'Laugh', 'Noise', 'Silence']

# دالة استخراج الميزات الصوتية من ملف صوتي
def extract_features(file_path, n_mfcc=13, duration=6, target_sr=22050):
    """
    استخراج MFCC وميزات إضافية مثل RMS و ZCR و Pitch من الملف الصوتي.
    """
    y, sr = librosa.load(file_path, sr=target_sr, duration=duration)
    
    # استخراج MFCC وحساب المتوسط
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    
    # استخراج RMS (جذر متوسط المربعات)
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    
    # استخراج ZCR (معدل عبور الصفر)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    
    # استخراج Pitch
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    
    # دمج جميع الميزات في مصفوفة واحدة
    return np.hstack((mfcc_mean, rms_mean, zcr_mean, pitch_mean))

# دالة تجهيز البيانات
def prepare_data(data_path, categories):
    X = []  # الميزات
    y = []  # الفئات (التصنيفات)
    
    for category in categories:
        folder_path = os.path.join(data_path, category)
        label = categories.index(category)
        
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(folder_path, file_name)
                try:
                    features = extract_features(file_path)
                    X.append(features)
                    y.append(label)
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
                    
    return np.array(X), np.array(y)

# تحميل وتجهيز البيانات
X, y = prepare_data(DATA_PATH, CATEGORIES)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# إضافة بُعد الزمن (timesteps) للمصفوفات لتناسب مدخلات LSTM
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

# تحويل الفئات إلى One-hot Encoding
y_train = to_categorical(y_train, num_classes=len(CATEGORIES))
y_test = to_categorical(y_test, num_classes=len(CATEGORIES))

# بناء النموذج باستخدام Sequential
model = Sequential()

# نبدأ باستخدام Input layer لتعريف شكل البيانات
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))

# طبقات LSTM
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.3))

model.add(BatchNormalization())

# الطبقة الأخيرة من LSTM تم تعديلها لإرجاع تسلسل (3 أبعاد)
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.3))

# طبقة GlobalAveragePooling1D لتجميع المعلومات عبر البُعد الزمني
model.add(GlobalAveragePooling1D())

# طبقة Dense مع تفعيل relu
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# الطبقة النهائية للتصنيف مع تفعيل softmax
model.add(Dense(len(CATEGORIES), activation='softmax'))

# تجميع النموذج باستخدام Adam optimizer ودالة الخسارة categorical_crossentropy
model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# تدريب النموذج
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# حفظ النموذج المدرب
model.save('cry_detection_model.h5')

# تقييم النموذج
loss, accuracy = model.evaluate(X_test, y_test)
print(f"The model accuracy: {accuracy * 100:.2f}%")

# رسم منحنيات الخسارة والدقة
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
