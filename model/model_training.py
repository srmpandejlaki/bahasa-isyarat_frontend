import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback

# === Konfigurasi ===
DATASET_DIR = 'dataset'
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 25
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

# === Kompilasi Model ===
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === Callback untuk Menampilkan Progres Training ===
class PrintAccuracyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)
        print(f"📚 Epoch {epoch+1}/{EPOCHS}")
        print(f"🔹 Train Accuracy: {acc:.4f} | Validation Accuracy: {val_acc:.4f}")

# === Training ===
print("\n🚀 Mulai training model...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[PrintAccuracyCallback()]
)

# === Hasil Akhir ===
final_val_acc = history.history['val_accuracy'][-1]
print(f"\n✅ Training selesai! Akurasi validasi akhir: {final_val_acc * 100:.2f}%")
