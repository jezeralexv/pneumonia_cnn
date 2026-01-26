import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Simple CNN skeleton
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("CNN model is built and compiled successfully!")

import tensorflow as tf

data_dir = r"C:\pneumonia project\pneumonia_cnn\archive\chest_xray"

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
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(lambda x, y: (x/255.0, y)).cache().prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.map(lambda x, y: (x/255.0, y)).cache().prefetch(buffer_size=AUTOTUNE)
test_ds  = test_ds.map(lambda x, y: (x/255.0, y)).cache().prefetch(buffer_size=AUTOTUNE)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

loss, acc = model.evaluate(test_ds)
print(f"Test accuracy: {acc:.2f}")