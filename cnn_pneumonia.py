import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ✅ Data directory
data_dir = r"C:\pneumonia project\pneumonia_cnn\archive\chest_xray"

# ✅ Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir + "/train",
    image_size=(150, 150),
    batch_size=32
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir + "/val",
    image_size=(150, 150),
    batch_size=32
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir + "/test",
    image_size=(150, 150),
    batch_size=32
)

# ✅ Normalize + Prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (x/255.0, y)).cache().prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.map(lambda x, y: (x/255.0, y)).cache().prefetch(buffer_size=AUTOTUNE)
test_ds  = test_ds.map(lambda x, y: (x/255.0, y)).cache().prefetch(buffer_size=AUTOTUNE)

# ✅ Data Augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ✅ Pretrained MobileNetV2 base
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(150,150,3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # freeze pretrained layers

# ✅ Final model
model = keras.Sequential([
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# ✅ Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ✅ Learning rate scheduler
lr_schedule = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6
)

# ✅ Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[lr_schedule]
)

# ✅ Evaluate
loss, acc = model.evaluate(test_ds)
print(f"Test accuracy: {acc:.2f}")

# ✅ Save in modern Keras 3 format
model.save("pneumonia_cnn_model.keras")

# ✅ Plot training history
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()