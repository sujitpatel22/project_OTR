import xml.etree.ElementTree as ET
import numpy as np
import cv2
import os
import sys

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


    def load_dataset_by_filname(self, data_dir, train_size = 0, val_size = 0, test_size = 0):
        total = train_size+val_size+test_size
        if(total > 1):
            train_size /= total
            val_size /= total
            test_size /= total

        train_size = int(self.input_size * train_size)
        val_size = int(self.input_size * val_size) + train_size
        test_size = int(self.input_size * test_size) + val_size
        self.train_data = (np.ones([train_size, self.img_height, self.img_width, 1]), [])
        self.val_data = (np.ones([val_size - train_size, self.img_height, self.img_width, 1]), [])
        self.test_data = (np.ones([test_size - val_size, self.img_height, self.img_width, 1]), [])
        index = 0
        for word_class in range(1470, 2199, 1):
            try:
                category_path = os.listdir(os.path.join(data_dir, str(word_class)))
            except:
                continue
            for category in category_path:
                filenames = os.listdir(os.path.join(data_dir, str(word_class), category))
                for filename in filenames:
                    file_path = os.path.join(data_dir, str(word_class), category, filename)
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    img = resize(img)
                    img = np.expand_dims(img, axis=-1)
                    img = img / 255
                    label = filename.split(".")[0].split("_")[1]
                    label = label.upper()
                    label = self.encode_label(label)
                    if index < train_size:
                        self.train_data[0][index] = img
                        self.train_data[1].append(label)
                    elif index < val_size:
                        self.val_data[0][index - train_size] = img
                        self.val_data[1].append(label)
                    elif index < test_size:
                        img=np.expand_dims(img, axis=0)
                        self.test_data[0][index - val_size] = img
                        self.test_data[1].append(label)
                    else:
                        return True
                    index += 1


    def load_dataset_by_xml(self, imgs_dir, labels_dir, train_size = 0, val_size = 0, test_size = 0):
        total = train_size+val_size+test_size
        if(total > 1):
            train_size /= total
            val_size /= total
            test_size /= total
            
        def extract_labels_by_xml(xml_path):
            with open(xml_path, "r") as file:
                xml_data = file.read()
                root = ET.fromstring(xml_data)
                word_tags = root.findall(".//word")
                extracted_labels = {word.get("id"): word.get("text") for word in word_tags}
                return extracted_labels

        train_size = int(self.input_size * train_size)
        val_size = int(self.input_size * val_size) + train_size
        test_size = int(self.input_size * test_size) + val_size
        self.train_data = (np.ones([train_size, self.img_height, self.img_width, 1]), [])
        self.val_data = (np.ones([val_size - train_size, self.img_height, self.img_width, 1]), [])
        self.test_data = (np.ones([test_size - val_size, self.img_height, self.img_width, 1]), [])

        index = 0
        for img_class in os.listdir(labels_dir):
            for xml_file in os.listdir(os.path.join(labels_dir, img_class)):
                extracted_labels = extract_labels_by_xml(os.path.join(labels_dir, img_class, xml_file))
                for file_name, label in extracted_labels.items():
                    img_category = xml_file.split(".")[0].split("-")
                    img_category = img_category[0] + "-" + img_category[1]

                    file_path = os.path.join(imgs_dir, img_class, img_category, file_name+".png")
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    label = label.upper()
                    label = self.encode_label(label)
                    if img is None or len(label) == 0 or -1 in label:
                        continue
                    img = self.resize(img)
                    img = np.expand_dims(img, axis=-1)
                    img = img / 255
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


    def load_dataset_by_csv(self, data_dir, train_dir, val_dir, test_dir, train_csv, val_csv, test_csv,  train_size = 0, val_size = 0, test_size = 0):
        total = train_size+val_size+test_size
        if(total > 1):
            train_size /= total
            val_size /= total
            test_size /= total

        train_size = int(self.input_size * train_size)
        val_size = int(self.input_size * val_size)
        test_size = int(self.input_size * test_size)
        self.train_data = (np.ones([train_size, self.img_height, self.img_width, 1]), [])
        self.val_data = (np.ones([val_size, self.img_height, self.img_width, 1]), [])
        self.test_data = (np.ones([test_size, self.img_height, self.img_width, 1]), [])

        data_limit = 0
        sub_dir = None
        for data_csv in [train_csv, val_csv, test_csv]:
            data_limit = train_size if data_csv == train_csv else val_size if data_csv == val_csv else test_size
            sub_dir = train_dir if data_csv == train_csv else val_dir if data_csv == val_csv else test_dir
            with open(os.path.join(data_dir, data_csv), 'r') as file:
                index = 0
                for line in file.readlines()[1:]:
                    if index >= data_limit:
                        break
                    parts = line.strip().split(',')
                    img_name, label = parts[0], parts[1]
                    img = cv2.imread(os.path.join(data_dir, sub_dir, img_name), cv2.IMREAD_GRAYSCALE)
                    label = label.upper()
                    label = self.encode_label(label)
                    if img is None or len(label) == 0 or -1 in label:
                        continue
                    try:
                        img = self.get_cropped(img)
                    except:
                        continue
                    img = self.resize(img)
                    # cv2.imwrite("temp/resized_temp.png", img)
                    # sys.exit()
                    img = np.expand_dims(img, axis=-1)
                    img = img / 255
                    if data_csv == train_csv:
                        self.train_data[0][index] = img
                        self.train_data[1].append(label)
                    elif data_csv == val_csv:
                        self.val_data[0][index] = img
                        self.val_data[1].append(label)
                    elif data_csv == test_csv:
                        self.test_data[0][index] = img
                        self.test_data[1].append(label)
                    index += 1


    def load_next_batch(self, data_cat):
        while True:
            images = np.ones([self.batch_size, self.img_height, self.img_width, 1])
            labels = np.ones([self.batch_size, self.max_label_length]) * -1
            input_length = np.ones((self.batch_size, 1)) * 40
            label_length = np.zeros((self.batch_size, 1))

            for i in range(self.batch_size):
                if data_cat == "train":
                    if self.train_ptr >= len(self.train_data[0]):
                        self.train_ptr = 0
                    img = self.train_data[0][self.train_ptr]
                    label = self.train_data[1][self.train_ptr]
                    self.train_ptr += 1
                elif data_cat == "val":
                    if self.val_ptr >= len(self.val_data[0]):
                        self.val_ptr = 0
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

    def encode_label(self, label):
        enc_label = []
        for char in label:
            enc_label.append(self.characters.find(char))
        return enc_label

    def resize(self, image):
        scale_x = self.img_width / image.shape[1]
        scale_y = self.img_height / image.shape[0]
        scale = min(scale_x, scale_y)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        pad_x = max(0, (self.img_width - image.shape[1]) // 2)
        pad_y = max(0, (self.img_height - image.shape[0]) // 2)
        image = cv2.copyMakeBorder(image, pad_y, self.img_height - image.shape[0] - pad_y,
                                   pad_x, self.img_width - image.shape[1] - pad_x, cv2.BORDER_CONSTANT, value=255)
        return image

    def get_cropped(self, img):
        _, img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_boxes = [cv2.boundingRect(contour) for contour in contours]

        x_min = min(box[0] for box in text_boxes)
        y_min = min(box[1] for box in text_boxes)
        x_max = max(box[0] + box[2] for box in text_boxes)
        y_max = max(box[1] + box[3] for box in text_boxes)

        cropped_image = img[y_min:y_max, x_min:x_max]
        # cv2.imwrite("temp/cropped_temp.png", cropped_image)
        _, img = cv2.threshold(cropped_image, 180, 255, cv2.THRESH_BINARY_INV)
        return img

        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmented_words = []
        img_num = 1
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            word_image = img[y:y+h, x:x+w]
            word_image = cv2.bitwise_not(word_image)
            segmented_words.append(word_image)
            cv2.imwrite(f"{image}_contour_{img_num}.png", word_image)
            img_num += 1