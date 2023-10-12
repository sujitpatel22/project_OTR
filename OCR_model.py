import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import os
import sys

IMG_WIDTH = 28
IMG_HEIGHT = 28

def main():
    data_dir = sys.argv[1]

    X_train, Y_train, X_test, Y_test = get_train_test_data(data_dir)

    Y_train = tf.keras.utils.to_categorical(Y_train)
    Y_test = tf.keras.utils.to_categorical(Y_test)

    model = build_model()
    model = compile_model(model)

    model.fit(X_train, Y_train, epochs = 20)

    model.evaluate(X_test, Y_test, verbose = 2)

    model.save("model_OCR")


def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation = "relu", input_shape = (IMG_WIDTH, IMG_HEIGHT, 1)),
        tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
        tf.keras.layers.Conv2D(32, (3,3), activation = "relu", input_shape = (IMG_WIDTH, IMG_HEIGHT, 1)),
        tf.keras.layers.MaxPooling2D(pool_size = (2,2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation = "relu"),

        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(32, activation = "softmax")
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
    # MNIST = tf.keras.datasets.mnist
    # (X_train_mnist, Y_train_mnist), (X_test_mnist,Y_test_mnist) = MNIST.load_data()
    images, labels = load_nist_data(data_dir)
    X_train_nist, X_test_nist, Y_train_nist,Y_test_nist = train_test_split(images, labels, test_size = 0.38, random_state=42)

    # X_train_nist = X_train_nist.reshape(X_train_nist.shape[0], X_train_nist.shape[1], X_train_nist.shape[2], 1)
    # X_test_nist = X_test_nist.reshape(X_test_nist.shape[0], X_test_nist.shape[1], X_test_nist.shape[2], 1)

    X_train_nist, X_test_nist = X_train_nist / 255.0, X_test_nist / 255.0

    # X_train = np.concatenate((X_train_mnist, X_train_nist), axis = 0)
    # Y_train = np.concatenate((Y_train_mnist, Y_train_nist), axis = 0)
    # X_test = np.concatenate((X_test_mnist, X_test_nist), axis = 0)
    # Y_test = np.concatenate((Y_test_mnist, Y_test_nist), axis = 0)

    return (X_train_nist, Y_train_nist, X_test_nist, Y_test_nist)


def load_nist_data(data_dir):
    images = []
    labels = []
    for hsf in os.listdir(data_dir):
        for field in os.listdir(os.path.join(data_dir, hsf)):
            for category in os.listdir(os.path.join(data_dir, hsf, field)):
                category_path = os.path.join(data_dir, hsf, field, category)
                filenames = os.listdir(os.path.join(data_dir, hsf, field, category))
                for filename in filenames:
                    img = cv2.imread(os.path.join(category_path, filename))
                    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img = np.expand_dims(img, axis=-1)
                    images.append(img)
                    labels.append(category)

    return (np.array(images), np.array(labels))


if __name__ == "__main__":
    main()