from tensorflow.keras.utils import Sequence
import numpy as np
import pickle
import random
import tensorflow as tf
import tensorflow.keras as keras


def preprocessing(batch_x, mode):
    frames_number = 32
    batch_x_images = []

    for paths in batch_x:
        stack_images = tf.io.decode_jpeg(tf.io.read_file(paths[0]), channels=1)

        for path in paths[1:]:
            image_string = tf.io.read_file(path)
            image = tf.io.decode_jpeg(image_string, channels=1)
            stack_images = tf.concat([stack_images, image], axis=-1)

        if stack_images.shape[-1] == frames_number:
            stack_images = tf.image.convert_image_dtype(stack_images, tf.float32)

        elif stack_images.shape[-1] > frames_number:
            # time augmentation
            if mode == 'train' or mode == 'val':
                diff = stack_images.shape[-1] - frames_number
                random_split = random.randint(0, diff)
                stack_images = tf.slice(stack_images, [0, 0, random_split], [-1, -1, frames_number])
            else:
                diff = stack_images.shape[-1] - frames_number
                avg_split = diff // 2
                stack_images = tf.slice(stack_images, [0, 0, avg_split], [-1, -1, frames_number])

        elif stack_images.shape[-1] < frames_number:
            for i in range(frames_number - stack_images.shape[-1]):
                stack_images = tf.concat([stack_images, image], axis=-1)

        stack_images = tf.image.convert_image_dtype(stack_images, tf.float32)
        batch_x_images.append(stack_images)

    return batch_x_images


class DataGenerator(Sequence):
    def __init__(self, data_info_path, batch_size, mode):
        self.batch_size = batch_size
        self.image_filenames = []
        self.labels = []
        self.n_classes = 51
        self.mode = mode

        with open(data_info_path, 'rb') as f:
            info = pickle.load(f)[self.mode]

        for image_filenames, label in info:
            self.image_filenames.append(image_filenames)
            self.labels.append(label)

    def shuffle_data(self):
        zip_data = list(zip(self.image_filenames, self.labels))
        random.shuffle(zip_data)
        self.image_filenames, self.labels = zip(*zip_data)

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        batch_x_images = preprocessing(batch_x, mode=self.mode)
        batch_x_images = np.asarray(batch_x_images)
        batch_y = keras.utils.to_categorical(batch_y, num_classes=51)

        return batch_x_images, batch_y
