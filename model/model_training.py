import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflowjs.converters import converter
from tqdm import tqdm
import subprocess

# === Konfigurasi ===
DATASET_DIR = 'dataset'       # ganti jika nama folder beda
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 15
H5_MODEL_PATH = 'bisindo_model.h5'
TFJS_MODEL_DIR = 'model'

# === Preprocessing dan Augmentasi ===
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# === Load Dataset ===
train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# === CNN Model ===
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === Callback: Progres Training + Simpan Model Terbaik ===
class TQDMCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n📚 Epoch {epoch+1}/{EPOCHS}")
    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)
        print(f"🔹 Train acc: {acc:.4f} | Val acc: {val_acc:.4f}")

checkpoint = ModelCheckpoint(H5_MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)

# === Training ===
print("\n🚀 Mulai training model...")
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[TQDMCallback(), checkpoint]
)

print(f"\n✅ Model .h5 berhasil disimpan: {H5_MODEL_PATH}")

# === Konversi ke TensorFlow.js ===
print("\n🔄 Mengonversi ke TensorFlow.js format...")
try:
    if not os.path.exists(TFJS_MODEL_DIR):
        os.makedirs(TFJS_MODEL_DIR)

    subprocess.run([
        'tensorflowjs_converter',
        '--input_format=keras',
        H5_MODEL_PATH,
        TFJS_MODEL_DIR
    ], check=True)

    print(f"✅ Model berhasil dikonversi ke folder: {TFJS_MODEL_DIR}")
    print("📁 File yang dihasilkan:")
    for file in os.listdir(TFJS_MODEL_DIR):
        print(f" - {file}")
except Exception as e:
    print(f"❌ Gagal konversi ke TensorFlow.js: {e}")
