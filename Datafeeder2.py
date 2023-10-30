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
            scale_x = self.img_width / image.shape[1]
            scale_y = self.img_height / image.shape[0]
            scale = min(scale_x, scale_y)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
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
                    label = encode_label(label)
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
