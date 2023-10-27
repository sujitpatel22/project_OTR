import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, Activation, Bidirectional, LSTM
from tensorflow.keras.layers import BatchNormalization, Dropout, Lambda, Reshape
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as BK
import cv2
import numpy as np
from itertools import groupby
from math import ceil
import os
import sys

def main():
    data_dir = sys.argv[1]

    DF = DataFeeder(170, 32, n = 40000, batch_size = 100, max_label_length = 18)
    print("loading dataset...")
    DF.load_dataset(data_dir, train_size = 0.50, val_size = 0.25, test_size = 0.25)
    print("Data loading done!")
    print("train_length = equal: ", len(DF.train_data[0]) == len(DF.train_data[1]))
    print("val_length = equal: ", len(DF.val_data[0]) == len(DF.val_data[1]))
    print("test_length = equal: ", len(DF.test_data[0]) == len(DF.test_data[1]))

    print("\nBuilding model...\n")
    MG = ModelGenerator(170, 32, len(DF.characters)+1, DF.max_label_length)
    crnn_model_input, Y_pred, crnn_model = MG.build_model("train")
    print(crnn_model.summary())

    test_func = BK.function([crnn_model_input], [Y_pred])
    viz_cb_train = VizCallback(test_func, img_fetcher = DF.load_next_batch(data_cat="train"), is_train = True, batch_count = ceil(len(DF.train_data)/DF.batch_size))
    viz_cb_val = VizCallback(test_func, img_fetcher = DF.load_next_batch(data_cat="val"), is_train = False, batch_count = ceil(len(DF.val_data)/DF.batch_size))

    print("compiling model...")
    crnn_model = MG.compile(crnn_model)
    print("training model...")
    crnn_model.fit_generator(generator = DF.load_next_batch(data_cat="train"),
                            steps_per_epoch = ceil(len(DF.train_data)/DF.batch_size),
                            epochs = 20, callbacks = [viz_cb_train,viz_cb_val],
                            validation_data = DF.load_next_batch(data_cat="val"),
                            validation_steps = ceil(len(DF.val_data)/DF.batch_size))

    crnn_model.save_weights("crnn_weights.h5")
    model = MG.build_model("save")
    model.load_weights("crnn_weights.h5")

    print("evaluating model...")
    word_accuracy, char_accuracy, model_accuracy = MG.evaluate(model, DF.test_data)
    print("word accuracy: ", word_accuracy)
    print("char accuracy: ", char_accuracy)
    print("model accuracy: ", model_accuracy)


class DataFeeder():
    def __init__(self, img_width = 146, img_height = 31, n = 100000, batch_size = 10000, max_label_length = 32):
        self.img_width = img_width
        self.img_height = img_height
        self.input_size = n
        self.batch_size = batch_size
        self.max_label_length = max_label_length
        self.characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.train_data = None
        self.val_data = None
        self.test_data = None   
        self.train_ptr = 0
        self.val_ptr = 0

    def load_dataset(self, data_dir, train_size = 0, val_size = 0, test_size = 0):
        total = train_size+val_size+test_size
        if(total > 1):
            train_size /= total
            val_size /= total
            test_size /= total
            
        def resize(image):
            if image.shape[1] > self.img_width or image.shape[0] > self.img_height:
                image = cv2.resize(image, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA)
            else:
                pad_x = max(0, (self.img_width - image.shape[1]) // 2)
                pad_y = max(0, (self.img_height - image.shape[0]) // 2)
                image = cv2.copyMakeBorder(image, pad_y, self.img_height - image.shape[0] - pad_y,
                pad_x, self.img_width - image.shape[1] - pad_x, cv2.BORDER_CONSTANT, value=255)
            return image
            
        def encode_label(label):
            enc_label = []
            for char in label:
                enc_label.append(self.characters.find(char))
            return enc_label

        train_size = int(self.input_size * train_size)
        val_size = int(self.input_size * val_size) + train_size
        test_size = int(self.input_size * test_size) + val_size
        self.train_data = (np.zeros([train_size, self.img_width, self.img_height, 1]), [])
        self.val_data = (np.zeros([val_size - train_size, self.img_width, self.img_height, 1]), [])
        self.test_data = (np.zeros([test_size - val_size, self.img_width, self.img_height, 1]), [])
        index = 0
        for word_class in range(1480, 1680, 1):
            for category in os.listdir(os.path.join(data_dir, str(word_class))):
                filenames = os.listdir(os.path.join(data_dir, str(word_class), category))
                for filename in filenames:
                    file_path = os.path.join(data_dir, str(word_class), category, filename)
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.img_width, self.img_height))
                    img = img.T
                    img = np.expand_dims(img, axis=-1)
                    img = img / 255
                    label = filename.split(".")[0].split("_")[1]
                    label = label.upper()
                    label = encode_label(label)
                    if index < train_size:
                        self.train_data[0][index] = img
                        self.train_data[1].append(label)
                    elif index < val_size:
                        self.val_data[0][index - train_size] = img
                        self.val_data[1].append(label)
                    elif index < test_size:
                        self.test_data[0][index - val_size] = img
                        self.test_data[1].append(label)
                    else:
                        return True
                    index += 1


    def load_next_batch(self, data_cat):
        while True:
            images = np.ones([self.batch_size, self.img_width, self.img_height, 1])
            labels = np.ones([self.batch_size, self.max_label_length]) * -1
            input_length = np.ones((self.batch_size, 1)) * 20
            label_length = np.zeros((self.batch_size, 1))

            for i in range(self.batch_size):
                if data_cat == "train":
                    # if self.train_ptr >= len(self.train_data[0]):
                    #     self.train_ptr = 0
                    img = self.train_data[0][self.train_ptr]
                    label = self.train_data[1][self.train_ptr]
                    self.train_ptr += 1
                elif data_cat == "val":
                    # if self.val_ptr >= len(self.val_data[0]):
                    #     self.val_ptr = 0
                    img = self.val_data[0][self.val_ptr]
                    label = self.val_data[1][self.val_ptr]
                    self.val_ptr += 1
                images[i] = img
                labels[i, 0: len(label)] = label
                label_length[i] = len(label)

            inputs = {
                "input_layer": images,
                "labels_layer": labels,
                "input_length_layer": input_length,
                "label_length_layer": label_length
            }
            outputs = {"ctc": np.zeros([self.batch_size])}
            yield (inputs, outputs)


class ModelGenerator():
    def __init__(self, img_width, img_height, n_classes, max_label_length):
        self.shape = (img_width, img_height, 1)
        self.n_classes = n_classes
        self.max_label_length = max_label_length

    def ctc_loss_gen(self, args):
        labels, Y_pred, input_length, label_length = args
        Y_pred = Y_pred[:, 2:, :]
        return BK.ctc_batch_cost(labels, Y_pred, input_length, label_length)

    def build_model(self, phase):
        model_input = Input(shape = self.shape, dtype="float32", name="input_layer")
        model = Conv2D(64, (3,3), activation = "relu", padding = "same", kernel_initializer="he_normal", name="conv1") (model_input)
        model = BatchNormalization() (model)
        model = MaxPooling2D(pool_size = (2, 2)) (model)

        model = Conv2D(128, (3,3), activation = "relu", padding = "same", kernel_initializer="he_normal", name="conv2") (model)
        model = BatchNormalization() (model)
        model = MaxPooling2D(pool_size = (2, 2)) (model)

        model = Conv2D(256, (3,3), activation = "relu", padding = "same", kernel_initializer="he_normal", name="conv3") (model)
        model = BatchNormalization() (model)
        model = Conv2D(256, (3,3), activation = "relu", padding = "same", kernel_initializer="he_normal", name="conv4") (model)
        model = Dropout(0.25) (model)
        model = BatchNormalization() (model)
        model = MaxPooling2D(pool_size = (1, 2)) (model)

        model = Conv2D(512, (3,3), activation = "relu", padding = "same", kernel_initializer="he_normal", name="conv5") (model)
        model = BatchNormalization() (model)
        model = Conv2D(512, (3,3), activation = "relu", padding = "same", kernel_initializer="he_normal", name="conv6") (model)
        model = Dropout(0.25) (model)
        model = BatchNormalization() (model)
        model = MaxPooling2D(pool_size = (1, 2)) (model)

        model = Conv2D(512, (2,2), activation = "relu", padding = "same", kernel_initializer="he_normal", name="conv7") (model)
        model = Dropout(0.25) (model)
        model = BatchNormalization() (model)
        
        reshape_input = (model.get_shape()[1], model.get_shape()[2] * model.get_shape()[3])
        model = Reshape(target_shape = reshape_input, name="reshape_layer") (model)
        model = Dense(64, activation = "relu", kernel_initializer="he_normal", name = "dense1") (model)
        model = Bidirectional(LSTM(256, return_sequences = True, kernel_initializer="he_normal"), merge_mode = "sum", name = "blstm1") (model)
        model = Bidirectional(LSTM(256, return_sequences = True, kernel_initializer="he_normal"), merge_mode = "concat", name = "blstm2") (model)

        labels = Input(shape=[self.max_label_length], dtype="int64", name="labels_layer")
        Y_pred = Dense(self.n_classes, activation="softmax", kernel_initializer="he_normal", name="output_layer_dense3") (model)
        input_length = Input(shape=[1], dtype="int64", name="input_length_layer")
        label_length = Input(shape=[1], dtype="int64", name="label_length_layer")

        ctc_loss = Lambda(self.ctc_loss_gen, output_shape=(1,), name = "ctc")([labels, Y_pred, input_length, label_length])
    
        if phase == "train":
            return model_input, Y_pred, Model(inputs= [model_input, labels, input_length, label_length], outputs = ctc_loss)
        elif phase == "save":
            model = Model(inputs= [model_input], outputs = Y_pred)
            return model
        else:
            sys.exit()
    

    def compile(self, model):
        model.compile(
            optimizer = Adam(),
            loss = {'ctc': lambda labels, y_pred: y_pred}
            )
        return model

    def evaluate(self, model, test_data):
        word_accuracy = 0
        char_accuracy = 0
        model_accuracy = 0
        chars = 0
        for img, label in zip(test_data[0], test_data[1]):
            img=np.expand_dims(img, axis=0)
            predicted_label = model.predict(img)
            predicted_label = self.decode_label(predicted_label)
            print("predicted: ", predicted_label)
            print("label: ", label)
            if predicted_label == label:
                word_accuracy += 1
            for i in range(min(len(predicted_label), len(label))):
                if predicted_label[i] == label[i]:
                    char_accuracy += 1
                chars += 1

        word_accuracy = word_accuracy/len(test_data[1]) * 100
        char_accuracy = char_accuracy/chars * 100
        model_accuracy = (word_accuracy + char_accuracy) / 2
        return word_accuracy, char_accuracy, model_accuracy

    def decode_label(self, label):
        label = list(np.argmax(label[0, 2:], axis = 1))
        label = [k for k, g in groupby(label)]
        return label

    def convert_label(self, label):
        string  = ""
        characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        for i in label:
            string += characters[i]
        return string


class VizCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_func, img_fetcher ,is_train, batch_count):
        self.test_func = test_func
        self.img_fetcher = img_fetcher
        self.is_train = is_train               
        self.batch_count = batch_count 

    def on_epoch_end(self, epoch, logs={}):
        self.show_accuracy_metrics(self.batch_count)

    def show_accuracy_metrics(self,num_batches):
        accuracy=0
        letter_accuracy=0
        batches_cnt=num_batches
        while batches_cnt>0:
            data_batch = next(self.img_fetcher)[0]
            decoded_labels = self.decode_batch(data_batch['input_layer'])
            actual_labels = data_batch['labels_layer']
            acc , let_acc = self.accuracies(actual_labels, decoded_labels)
            accuracy += acc
            letter_accuracy += let_acc
            batches_cnt -= 1

        accuracy = accuracy/num_batches
        letter_accuracy = letter_accuracy/num_batches
        if self.is_train:
            print(f"\nTrain Average Accuracy of {str(num_batches)} Batches: {np.round(accuracy,2)} %")
            print(f"Train Average Letter Accuracy of {str(num_batches)} Batches: {np.round(letter_accuracy,2)} %")
        else:
            print(f"Validation Average Accuracy of {str(num_batches)} Batches: {np.round(accuracy,2)} %")
            print(f"Validation Average Letter Accuracy of {str(num_batches)} Batches: {np.round(letter_accuracy,2)} %")
    
    def decode_batch(self, word_batch):
        out = self.test_func([word_batch])[0]
        ret = []
        for j in range(out.shape[0]):
            out_best = list(np.argmax(out[j, 2:], 1))
            out_best = [k for k, g in groupby(out_best)]
            ret.append(out_best)
        return ret

    def accuracies(self, labels, Y_pred):
        accuracy=0
        letter_acc=0
        letter_cnt=0
        count=0
        for i in range(len(labels)):
            predicted_output = self.decode_label(Y_pred[i])
            def extract_label(label):
                temp = []
                for i in label:
                    if i >= 0:
                        temp.append(i)
                return temp
            actual_output = extract_label(labels[i])
            actual_output = self.decode_label(actual_output)
            count += 1
            for j in range(min(len(predicted_output),len(actual_output))):
                if predicted_output[j] == actual_output[j]:
                    letter_acc += 1
            letter_cnt += max(len(predicted_output),len(actual_output))
            if actual_output == predicted_output:
                accuracy += 1
        final_accuracy = np.round((accuracy/len(labels))*100,2)
        final_letter_acc = np.round((letter_acc/letter_cnt)*100,2)
        return final_accuracy,final_letter_acc

    def decode_label(self, label):
        string  = ""
        characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        length = 36
        for i in label:
            if i < length:
                string += characters[int(i)]
        return string


if __name__ == "__main__":
    main()
