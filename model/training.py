import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# --- 1. Persiapan Data ---
dataset_dir = "dataset"

if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"Direktori {dataset_dir} tidak ditemukan!")

class_names = sorted([
    d for d in os.listdir(dataset_dir)
    if os.path.isdir(os.path.join(dataset_dir, d))
])

if len(class_names) == 0:
    raise ValueError("Tidak ada folder kelas ditemukan di direktori dataset!")

all_images = []
all_labels = []

print("Distribusi gambar per kelas:")
for class_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(dataset_dir, class_name)
    images = [
        os.path.join(class_dir, img_name)
        for img_name in os.listdir(class_dir)
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    print(f"Kelas {class_name}: {len(images)} gambar")
    if len(images) == 0:
        raise ValueError(f"Kelas {class_name} tidak memiliki gambar.")
    all_images.extend(images)
    all_labels.extend([class_idx] * len(images))

# Minimal 5 gambar per kelas, split train-test 80:20
train_images, train_labels = [], []
test_images, test_labels = [], []

for class_idx in range(len(class_names)):
    class_images = [img for img, lbl in zip(all_images, all_labels) if lbl == class_idx]
    if len(class_images) < 5:
        raise ValueError(f"Kelas {class_names[class_idx]} hanya punya {len(class_images)} gambar. Minimal 5 diperlukan.")
    tr_imgs, ts_imgs = train_test_split(class_images, test_size=0.2, random_state=42)
    train_images.extend(tr_imgs)
    train_labels.extend([class_idx] * len(tr_imgs))
    test_images.extend(ts_imgs)
    test_labels.extend([class_idx] * len(ts_imgs))

print(f"Jumlah gambar pelatihan: {len(train_images)}")
print(f"Jumlah gambar test: {len(test_images)}")

# --- 2. Fungsi Load & Preprocess ---
def load_and_preprocess_image(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Gambar tidak bisa dibaca: {image_path}")
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return img

# --- 3. Data Generator ---
def create_generator(image_paths, labels, batch_size=32, is_training=True):
    num_classes = len(class_names)
    while True:
        indices = np.random.permutation(len(image_paths)) if is_training else np.arange(len(image_paths))
        for start in range(0, len(image_paths), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_images, batch_labels = [], []
            for idx in batch_idx:
                try:
                    img = load_and_preprocess_image(image_paths[idx])
                except ValueError:
                    continue
                if is_training:
                    if np.random.rand() < 0.5:
                        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    if np.random.rand() < 0.5:
                        img = img * np.random.uniform(0.8, 1.2)
                        img = np.clip(img, 0, 1)
                batch_images.append(img)
                batch_labels.append(tf.keras.utils.to_categorical(labels[idx], num_classes=num_classes))
            if len(batch_images) == 0:
                continue
            yield np.array(batch_images, dtype=np.float32), np.array(batch_labels, dtype=np.float32)

train_gen = create_generator(train_images, train_labels, batch_size=32, is_training=True)
test_gen = create_generator(test_images, test_labels, batch_size=32, is_training=False)

# --- 4. Build Model (Transfer Learning MobileNetV2) ---
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
for layer in base_model.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(len(class_names), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- 5. Training ---
history = model.fit(
    train_gen,
    steps_per_epoch=max(1, len(train_images) // 32),
    validation_data=test_gen,
    validation_steps=max(1, len(test_images) // 32),
    epochs=30,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)

# --- 6. Simpan info ---
with open('train_test_split_info.json', 'w') as f:
    json.dump({
        'train_count': len(train_images),
        'test_count': len(test_images),
        'classes': class_names
    }, f)
print("Informasi pembagian data disimpan.")

# --- 7. Visualisasi ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Akurasi')
plt.plot(history.history['val_accuracy'], label='Val Akurasi')
plt.title('Accuracy')
plt.legend()
plt.show()

# --- 8. Evaluasi Model ---
y_true, y_pred = [], []
batch_size = 32
for i in range(0, len(test_images), batch_size):
    batch_imgs = test_images[i:i+batch_size]
    batch_labels = test_labels[i:i+batch_size]
    X_batch, _ = next(create_generator(batch_imgs, batch_labels, batch_size=len(batch_imgs), is_training=False))
    preds = model.predict(X_batch)
    y_true.extend(batch_labels)
    y_pred.extend(np.argmax(preds, axis=1))

print("Akurasi:", np.mean(np.array(y_true) == np.array(y_pred)))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("Classification Report:\n", classification_report(y_true, y_pred, target_names=class_names))

# --- 9. Simpan model & mapping kelas ---
model.save('model/training_results/bisindo_model.h5')
with open('model/training_results/class_indices.json', 'w') as f:
    json.dump({str(i): name for i, name in enumerate(class_names)}, f)
print("Model & mapping kelas disimpan.")
