# Kaggle Competition: Planet
# Created by Qixun Qu
# quqixun@gmail.com


import os
import cv2
from tqdm import *
import numpy as np
import pandas as pd
import imutils as iu
import tensorflow as tf


class PlanetTFRecords():

    IMG_AUG = 6
    IMAGESET_SIZE_256 = [256, 256, 3]
    IMAGESET_SIZE_64 = [64, 64, 3]
    IMAGESET_SIZE_32 = [32, 32, 3]
    PRINT_FREQ = 100

    def __init__(self):
        self.names = []
        self.labels = []
        self.train_idx = []
        self.valid_idx = []
        self.labels_num = 0
        self.train_num = 0
        self.valid_num = 0

        return

    def read_labels(self, path, train_prop=0.75, mode='data'):
        num_labels = pd.read_csv(path)
        data_num = num_labels.shape[0]
        self.names = num_labels['image_name'].tolist()
        self.labels = num_labels.drop('image_name', 1).as_matrix()
        self.labels_num = self.labels.shape[1]

        self.train_num = int(np.round(data_num * train_prop))
        self.valid_num = int(data_num - self.train_num)

        if mode == 'data':
            rnd_idx = np.random.permutation(data_num)
            self.train_idx = rnd_idx[:self.train_num]
            self.valid_idx = rnd_idx[self.train_num:]

        return

    def img_augment(self, img, img_size=64):
        img_aug = np.zeros((self.IMG_AUG, img_size, img_size, 3))

        img_aug[0, :, :, :] = img
        img_aug[1, :, :, :] = cv2.flip(img, 0)
        img_aug[2, :, :, :] = cv2.flip(img, 1)
        img_aug[3, :, :, :] = iu.rotate(img, 90)
        img_aug[4, :, :, :] = iu.rotate(img, 180)
        img_aug[5, :, :, :] = iu.rotate(img, 270)

        img_aug = img_aug.astype(np.uint8)

        return img_aug

    def create_tfrecords(self, image_size, tfr_path, dir_path, tfr='train'):
        if os.path.isfile(tfr_path):
            return

        if tfr == 'train':
            print("Creating TFRecords for training set:")
            idx = self.train_idx
        elif tfr == 'valid':
            print("Creating TFRecords for validating set:")
            idx = self.valid_idx
        else:
            print("Wrong TFRecords type, 'train' or 'valid'.")
            return

        writer = tf.python_io.TFRecordWriter(tfr_path)

        for i in tqdm(range(len(idx))):
            img_idx = idx[i]
            img_path = dir_path + self.names[img_idx] + '.jpg'
            img = cv2.imread(img_path)
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)

            img_aug = self.img_augment(img)
            for ia in range(self.IMG_AUG):
                img_raw = img_aug[ia, :, :, :].tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': tf.train.Feature(
                        float_list=tf.train.FloatList(value=self.labels[img_idx])),
                    'image': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[img_raw]))
                }))

                writer.write(example.SerializeToString())

        writer.close()

        return

    def decode_tfrecord(self, queue, img_size=64):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(queue)

        features = tf.parse_single_example(
            serialized_example,
            features={
                'label': tf.FixedLenFeature([self.labels_num], tf.float32),
                'image': tf.FixedLenFeature([], tf.string)
            })

        img = tf.decode_raw(features['image'], tf.uint8)
        img = tf.reshape(img, [img_size, img_size, 3])
        img = tf.cast(img, tf.float32) / 255.

        label = tf.cast(features['label'], tf.float32)

        return img, label

    def inputs(self, path, img_size, batch_size, epoch_num, mad):
        if not epoch_num:
            epoch_num = 1

        with tf.name_scope('input'):
            queue = tf.train.string_input_producer([path], num_epochs=epoch_num)
            image, label = self.decode_tfrecord(queue, img_size)

        images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=2,
                                                min_after_dequeue=mad, capacity=mad + 3 * batch_size)

        return images, labels


if __name__ == '__main__':
    '''
    Basic Settings
    '''

    # Create instance
    ptfr = PlanetTFRecords()

    # Set the path of the directory which contains images
    dir_path = os.getcwd() + '\\Dataset\\train-jpg\\'

    # Load labels and generate index for training data and
    # validating data respectively
    nl_path = os.getcwd() + '\\Dataset\\train_v2n.csv'
    ptfr.read_labels(nl_path, 0.75, 'data')

    # Settings
    image_size = 64  # or 32
    train_tfr = os.getcwd() + '\\TFRecords\\train_' + str(image_size) + '.tfrecords'
    valid_tfr = os.getcwd() + '\\TFRecords\\valid_' + str(image_size) + '.tfrecords'

    # Create TFRecords for training data
    ptfr.create_tfrecords(image_size, train_tfr, dir_path, 'train')

    # Create TFRecords for validating data
    ptfr.create_tfrecords(image_size, valid_tfr, dir_path, 'valid')
