import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import os
import sys

IMG_WIDTH = 28
IMG_HEIGHT = 28
EPOCHS = 10

def main():
    data_dir = sys.argv[1]

    X_train, Y_train, X_test, Y_test = get_train_test_data(data_dir)
    print("done loading dataset!")

    model = build_model()
    model = compile_model(model)

    model.fit(X_train, Y_train, epochs = EPOCHS)

    model.evaluate(X_test, Y_test, verbose = 2)

    model.save("model_OCR")


def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation = "relu", input_shape = (IMG_WIDTH, IMG_HEIGHT, 1)),
        tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
        # tf.keras.layers.Conv2D(64, (3,3), activation = "relu", input_shape = (IMG_WIDTH, IMG_HEIGHT, 1)),
        # tf.keras.layers.MaxPooling2D(pool_size = (2,2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(246, activation = "relu"),
        tf.keras.layers.Dense(246, activation = "relu"),
        # tf.keras.layers.Dense(273, activation = "relu"),

        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(123, activation = "softmax")
    ])
    return model


def compile_model(model):
    model.compile(
        optimizer = "adam",
        loss = "categorical_crossentropy",
        metrics = ["accuracy"]
    )
    return model


def get_train_test_data(data_dir):
    images, labels = load_nist_data(data_dir)

    labels = tf.keras.utils.to_categorical(labels)
    # print(len(labels))
    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size = 0.38)

    X_train, X_test = X_train / 255.0, X_test / 255.0
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    return (X_train, Y_train, X_test, Y_test)


def load_nist_data(data_dir):
    images = []
    labels = []
    for category in range(97, 123, 1):
        for hsf in os.listdir(os.path.join(data_dir, str(category))):
            filenames = os.listdir(os.path.join(data_dir, str(category), hsf))
            for filename in filenames:
                file_path = os.path.join(data_dir, str(category), hsf, filename)
                img = cv2.imread(file_path)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = np.expand_dims(img, axis=-1)
                images.append(img)
                labels.append(category)

    return (np.array(images), np.array(labels))


if __name__ == "__main__":
    main()