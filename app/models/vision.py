import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf  # For tf.data
import keras
from keras import layers
from keras.applications import EfficientNetB0
import os

IMG_SIZE = 224
BATCH_SIZE = 64

def init_model():
    current_directory = os.getcwd()
    print(current_directory)

    dataset_name = "beans"
    ds_info = tfds.builder(dataset_name).info

    def build_model(num_classes=120):
        inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

        # Freeze the pretrained weights
        model.trainable = False

        # Rebuild top
        x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
        x = layers.BatchNormalization()(x)

        top_dropout_rate = 0.2
        x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

        # Compile
        model = keras.Model(inputs, outputs, name="EfficientNet")
        optimizer = keras.optimizers.Adam(learning_rate=1e-2)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )
        return model

    model = build_model(3)
    model.load_weights("app/models/model.h5")

    return model, ds_info

def predict_model(image, model, ds_info):
    image = tf.cast(image, tf.float32)

    if image.shape[-1] == 4:
        image = image[..., :3]

    image = image
    size = (IMG_SIZE, IMG_SIZE)
    image = tf.image.resize(image, size)
    image = tf.expand_dims(image, 0)

    prediction = model.predict(image)

    label_info = ds_info.features["label"]

    for i in range(3):
        print(f"Es un {label_info.int2str(np.argsort(prediction)[0][-(i+1)])} con una probabilidad de: {np.sort(prediction)[0][-(i+1)]:2f}")

    return label_info.int2str(np.argsort(prediction)[0][-1])