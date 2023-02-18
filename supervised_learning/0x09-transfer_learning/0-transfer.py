#!/usr/bin/env python3
"""
Trains a convolutional neural network to classify the CIFAR 10 dataset.
"""

import tensorflow as tf
import tensorflow.keras as K  # type: ignore


def preprocess_data(X, Y):
    X_p = X.astype("float32")
    X_p /= 255.0
    Y_p = K.utils.to_categorical(Y, num_classes=10)
    return X_p, Y_p


def train_model():
    # Load the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = K.datasets.cifar10.load_data()

    # Preprocess the data
    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)

    # Define the base model
    input_tensor = K.layers.Input(shape=(32, 32, 3))
    x = K.layers.Lambda(
        lambda image: tf.image.resize(image, (299, 299)),
    )(input_tensor)
    base_model = K.applications.InceptionV3(
        weights="imagenet", include_top=False, input_tensor=x
    )

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add the top layers for classification
    x = base_model.output
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dense(512, activation="relu")(x)
    x = K.layers.Dense(256, activation="relu")(x)
    predictions = K.layers.Dense(10, activation="softmax")(x)
    model = K.models.Model(inputs=base_model.input, outputs=predictions)

    checkpoint = K.callbacks.ModelCheckpoint(
        "cifar10.h5", monitor="val_acc", save_best_only=True
    )

    # Compile the model
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint],
    )

    # Save the model
    model.summary()
    model.save("cifar10.h5")


if __name__ == "__main__":
    train_model()
