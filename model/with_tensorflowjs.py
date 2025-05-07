import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
# from tensorflowjs.converters import converter
from tqdm import tqdm
import subprocess
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# === Konfigurasi ===
DATASET_DIR = 'dataset'       # ganti jika nama folder beda
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 25
H5_MODEL_PATH = 'bisindo_model.h5'
TFJS_MODEL_DIR = 'model'
LEARNING_RATE = 0.001

# === Preprocessing dan Augmentasi ===
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 80% training, 20% validation
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

# Custom metrics
def precision(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=1)
    y_true = tf.argmax(y_true, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))

def recall(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=1)
    y_true = tf.argmax(y_true, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

# Compile model with custom learning rate and metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', precision, recall, f1]
)

# === Callback: Progres Training + Simpan Model Terbaik ===
class TQDMCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n📚 Epoch {epoch+1}/{EPOCHS}")
    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)
        prec = logs.get('precision', 0)
        rec = logs.get('recall', 0)
        f1_score = logs.get('f1', 0)
        print(f"🔹 Train acc: {acc:.4f} | Val acc: {val_acc:.4f}")
        print(f"🔹 Precision: {prec:.4f} | Recall: {rec:.4f} | F1-score: {f1_score:.4f}")

checkpoint = ModelCheckpoint(H5_MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)

# === Training ===
print("\n🚀 Mulai training model...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[TQDMCallback(), checkpoint]
)

print(f"\n✅ Model .h5 berhasil disimpan: {H5_MODEL_PATH}")

# === Evaluasi Model ===
print("\n📊 Evaluasi Model...")
# Get predictions
val_predictions = model.predict(val_data)
val_pred_classes = np.argmax(val_predictions, axis=1)
val_true_classes = val_data.classes

# Calculate metrics
precision = precision_score(val_true_classes, val_pred_classes, average='weighted')
recall = recall_score(val_true_classes, val_pred_classes, average='weighted')
f1 = f1_score(val_true_classes, val_pred_classes, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(val_true_classes, val_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.close()

print("\n✅ Confusion matrix telah disimpan sebagai 'confusion_matrix.png'")

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
