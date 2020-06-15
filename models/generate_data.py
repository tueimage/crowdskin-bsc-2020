import numpy as np
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import h5py


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
        count = 0
        hf = h5py.File(self.directory + 'all_images.h5', 'r')
        for filename in batch_x_in:
            # load_h5
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

            else:
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


def generate_data_2(directory, augmentation, batch_size, file_list, label_1, label_2, sample_weights,
                    sample_weights_3=None, sample_weights_4=None, label_3=None, label_4=None):
    i = 0
    image_batch_r = []
    label_1_batch_r = []
    label_2_batch_r = []
    sample_weight_r = []
    if label_3 is not None:
        label_3_batch_r = []
        sample_weight_3_r = []
    if label_4 is not None:
        label_4_batch_r = []
        sample_weight_4_r = []
    while True:
        image_batch = []
        label_1_batch = []
        label_2_batch = []
        sample_weight = []
        if label_3 is not None:
            label_3_batch = []
            sample_weight_3 = []
        if label_4 is not None:
            label_4_batch = []
            sample_weight_4 = []
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
            if label_3 is not None:
                label_3_batch.append(label_3[i])
                sample_weight_3.append(sample_weights_3[i])
            if label_4 is not None:
                label_4_batch.append(label_4[i])
                sample_weight_4.append(sample_weights_4[i])
            # print(str(label_1.iloc[i]) + ',' + str(label_2[i]) + ',' + str(sample_weights[i]) ) #debug

            i = i + 1
        hf.close()
        #check if all samples are zero
        sw_0 = all(sample == 0 for sample in sample_weight)
        if label_3 is not None:
            sw_3_0 = all(sample == 0 for sample in sample_weight_3)
        if label_4 is not None:
            sw_4_0 = all(sample == 0 for sample in sample_weight_4)
        # Return 2 labels
        # Memory, when all the samples for asymmetry score is zero generator returns previous batch instead of zeros
        if label_3 is None and label_4 is None:
            if sw_0:
                print('all weights zero')
                yield (
                    np.asarray(image_batch_r),
                    dict(out_class=np.asarray(label_1_batch_r), out_asymm=np.asarray(label_2_batch_r)),
                    dict(out_asymm=np.asarray(sample_weight_r)))
            else:
                image_batch_r = image_batch
                label_1_batch_r = label_1_batch
                label_2_batch_r = label_2_batch
                sample_weight_r = sample_weight
                yield (
                    np.asarray(image_batch),
                    dict(out_class=np.asarray(label_1_batch), out_asymm=np.asarray(label_2_batch)),
                    dict(out_asymm=np.asarray(sample_weight)))

        # return 3 labels
        elif label_3 is not None and label_4 is None:
            if sw_0 and sw_3_0:
                print('all weights zero')
                yield (
                    np.asarray(image_batch_r),
                    dict(out_class=np.asarray(label_1_batch_r), label_2=np.asarray(label_2_batch_r),
                         label_3=np.asarray(label_3_batch_r)),
                    dict(label_2=np.asarray(sample_weight_r), label_3=np.asarray(sample_weight_3_r)))
            else:
                image_batch_r = image_batch
                label_1_batch_r = label_1_batch
                label_2_batch_r = label_2_batch
                label_3_batch_r = label_3_batch
                sample_weight_r = sample_weight
                sample_weight_3_r = sample_weight_3
                yield (
                    np.asarray(image_batch),
                    dict(out_class=np.asarray(label_1_batch), label_2=np.asarray(label_2_batch),
                         label_3=np.asarray(label_3_batch)),
                    dict(label_2=np.asarray(sample_weight), label_3=np.asarray(sample_weight_3)))

        # Return 4 labels
        elif label_3 is not None and label_4 is not None:
            if sw_0 and sw_3_0 and sw_4_0:
                print('all weights zero')
                yield (
                    np.asarray(image_batch_r),
                    dict(out_class=np.asarray(label_1_batch_r), label_2=np.asarray(label_2_batch_r),
                         label_3=np.asarray(label_3_batch_r), label_4=np.asarray(label_4_batch_r)),
                    dict(label_2=np.asarray(sample_weight_r), label_3=np.asarray(sample_weight_3_r),
                         label_4=np.asarray(sample_weight_4_r)))
            else:
                image_batch_r = image_batch
                label_1_batch_r = label_1_batch
                label_2_batch_r = label_2_batch
                label_3_batch_r = label_3_batch
                label_4_batch_r = label_4_batch
                sample_weight_r = sample_weight
                sample_weight_3_r = sample_weight_3
                sample_weight_4_r = sample_weight_4
                yield (
                    np.asarray(image_batch),
                    dict(out_class=np.asarray(label_1_batch), label_2=np.asarray(label_2_batch),
                         label_3=np.asarray(label_3_batch), label_4=np.asarray(label_4_batch)),
                    dict(label_2=np.asarray(sample_weight), label_3=np.asarray(sample_weight_3),
                         label_4=np.asarray(sample_weight_4)))
        else:
            print()
            raise


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

        sample_weight_check = self.sample_weights[idx * self.batch_size:(idx + 1) * self.batch_size]

        sample_count = 0
        if all(sample == 0 for sample in sample_weight_check):
            while all(sample == 0 for sample in sample_weight_check):
                print('\r' + str(idx))
                print('\r' + str(sample_count))
                sample_count += 1
                sample_weight_check = self.sample_weights[(idx-sample_count) * self.batch_size:(idx-sample_count+1) * self.batch_size]
            # use previous batch if all weights are 0
            print('all weights zero')
        loc = idx - sample_count
        batch_x_in = self.file_list[loc * self.batch_size:(loc + 1) * self.batch_size]
        batch_y1 = self.label1[loc * self.batch_size:(loc + 1) * self.batch_size].to_numpy()
        batch_y2 = self.label2[loc * self.batch_size:(loc + 1) * self.batch_size]
        sample_weight = self.sample_weights[loc * self.batch_size:(loc + 1) * self.batch_size]
        print('\r' + str(loc))
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
        return (
            batch_x,
            {'out_class': batch_y1, 'out_asymm': batch_y2},
            {'out_asymm': sample_weight})
