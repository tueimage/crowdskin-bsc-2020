import numpy as np
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import h5py
from albumentations import Compose, ShiftScaleRotate, HorizontalFlip, VerticalFlip, \
    RGBShift, ToFloat, GridDistortion, Normalize
from cv2.xphoto import createGrayworldWB
from cv2 import BORDER_REPLICATE


class Generate_Alt_1(Sequence):

    def __init__(self, directory, augmentation, batchsize, file_list, label_1):
        self.file_list = file_list
        self.directory = directory
        self.label1 = label_1
        self.batch_size = batchsize
        self.augmentation = augmentation

    def __len__(self):
        return int(np.ceil((len(self.file_list) / float(self.batch_size))))

    def __getitem__(self, idx):
        batch_x_in = self.file_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.label1[idx * self.batch_size:(idx + 1) * self.batch_size].to_numpy()

        batch_length = len(batch_x_in)
        batch_x = np.empty((batch_length, 384, 384, 3))
        # batch_y = np.empty(batch_length)
        count = 0
        hf = h5py.File(self.directory + 'all_images.h5', 'r')
        for filename in batch_x_in:
            # load_h5
            img = hf.get(filename + '.jpg')
            img = np.array(img)

            if self.augmentation:
                # wb = createGrayworldWB()
                # wb.setSaturationThreshold(1)
                # img = wb.balanceWhite(img)
                datagen = ImageDataGenerator(
                    rotation_range=360,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.2,
                    zoom_range=0.2,
                    channel_shift_range=20,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode="nearest")

                batch_x[count] = datagen.random_transform(img) / 255.0


                # augmentations = Compose([GridDistortion(),
                #                          HorizontalFlip(),
                #                          VerticalFlip(),
                #                          ShiftScaleRotate(shift_limit=0.1, rotate_limit=360, scale_limit=0.1,
                #                                           border_mode=BORDER_REPLICATE, always_apply=True),
                #                          ])

                # batch_x[count, :, :, :] = augmentations(image=img)['image'] / 255.0
            else:
                # wb = createGrayworldWB()
                # wb.setSaturationThreshold(1)
                # img = wb.balanceWhite(img)
                batch_x[count, :, :, :] = img / 255.0
            count += 1
        # print('\r' + str(idx))  # debug
        hf.close()
        return batch_x, batch_y



def generate_data_1(directory, augmentation, batchsize, file_list, label_1):
    i = 0
    while True:
        image_batch = []
        label_1_batch = []
        hf = h5py.File(directory + 'all_images.h5', 'r')
        for b in range(batchsize):
            if i == (len(file_list)):
                i = 0
            img = hf.get(file_list.iloc[i] + '.jpg')
            img = np.array(img)
            # img = image.load_img(directory + file_list.iloc[i] + '.jpg', grayscale=False, target_size=(384, 384))
            # img = image.img_to_array(img)

            if augmentation:
                datagen = ImageDataGenerator(
                    rotation_range=360,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.2,
                    zoom_range=0.2,
                    channel_shift_range=20,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode="nearest")
                img = datagen.random_transform(img)
                img = img / 255.0
            else:
                img = img / 255.0

            image_batch.append(img)
            label_1_batch.append(label_1.iloc[i])
            i = i + 1
        hf.close()
        yield np.asarray(image_batch), np.asarray(label_1_batch)


def generate_data_2(directory, augmentation, batch_size, file_list, label_1, label_2, sample_weights):
    i = 0
    image_batch_r = []
    label_1_batch_r = []
    label_2_batch_r = []
    sample_weight_r = []
    while True:
        image_batch = []
        label_1_batch = []
        label_2_batch = []
        sample_weight = []
        hf = h5py.File(directory + 'all_images.h5', 'r')
        for b in range(batch_size):
            if i == (len(file_list)):
                i = 0
            img = hf.get(file_list.iloc[i] + '.jpg')
            img = np.array(img)

            if augmentation:
                datagen = ImageDataGenerator(
                    rotation_range=360,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.2,
                    zoom_range=0.2,
                    channel_shift_range=20,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode="nearest")
                img = datagen.random_transform(img)
                img = img / 255.0

            else:
                img = img / 255.0

            image_batch.append(img)
            label_1_batch.append(label_1.iloc[i])
            label_2_batch.append(label_2[i])
            sample_weight.append(sample_weights[i])

            # print(str(label_1.iloc[i]) + ',' + str(label_2[i]) + ',' + str(sample_weights[i]) )

            i = i + 1
            # print('\r', str(i/20))

        hf.close()
        if all(sample == 0 for sample in sample_weight):
            print('all weights zero')
            yield (
                np.asarray(image_batch_r),
                ({'out_class': np.asarray(label_1_batch_r), 'out_asymm': np.asarray(label_2_batch_r)}),
                ({'out_asymm': np.asarray(sample_weight_r)}))
        else:
            image_batch_r = image_batch  # Memory, when all the samples for asymmetry score is zero generator returns previous batch instead of zeros
            label_1_batch_r = label_1_batch
            label_2_batch_r = label_2_batch
            sample_weight_r = sample_weight
            yield (
                np.asarray(image_batch),
                {'out_class': np.asarray(label_1_batch), 'out_asymm': np.asarray(label_2_batch)},
                {'out_asymm': np.asarray(sample_weight)})


class Generate_Alt_2(Sequence):
    def __init__(self, directory, augmentation, batch_size, file_list, label_1, label_2, sample_weights, class_weights):
        self.file_list = file_list
        self.directory = directory
        self.label1 = label_1
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.label2 = label_2
        self.sample_weights = sample_weights
        self.class_weights = class_weights

    def __len__(self):
        return int(np.ceil(len(self.file_list) / float(self.batch_size)))

    def __getitem__(self, idx):

        sample_weights_in_check = self.sample_weights[idx * self.batch_size:(idx + 1) * self.batch_size]

        if all(sample == 0 for sample in sample_weights_in_check):
            # use previous batch if all weights are 0
            print('all weights zero')
            batch_x_in = self.file_list[(idx - 1) * self.batch_size:idx * self.batch_size]
            batch_y1 = self.label1[(idx - 1) * self.batch_size:idx * self.batch_size].to_numpy()
            batch_y2 = self.label2[(idx - 1) * self.batch_size:idx * self.batch_size]
            sample_weight = self.sample_weights[(idx - 1) * self.batch_size:idx * self.batch_size]
        else:
            batch_x_in = self.file_list[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y1 = self.label1[idx * self.batch_size:(idx + 1) * self.batch_size].to_numpy()
            batch_y2 = self.label2[idx * self.batch_size:(idx + 1) * self.batch_size]
            sample_weight = self.sample_weights[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_length = len(batch_x_in)
        batch_x = np.empty((batch_length, 384, 384, 3))

        count = 0
        hf = h5py.File(self.directory + 'all_images.h5', 'r')

        for filename in batch_x_in:
            # load h5
            img = hf.get(filename + '.jpg')
            img = np.array(img)

            if self.augmentation:
                datagen = ImageDataGenerator(
                    rotation_range=360,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.2,
                    zoom_range=0.2,
                    channel_shift_range=20,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode="nearest")
                batch_x[count] = datagen.random_transform(img) / 255.0

                batch_x[count, :, :, :] = img / 255.0

            else:
                batch_x[count, :, :, :] = img / 255.0

            count += 1
        hf.close()
        # print('\r' + str(idx))
        return (
            batch_x,
            {'out_class': batch_y1, 'out_asymm': batch_y2},
            {'out_asymm': sample_weight})
