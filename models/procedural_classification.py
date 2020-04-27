# RUN FLAGS
TRIAL = False
SANITY_CHECK = False
DEBUG = False
VERBOSE = True
GOOGLE_CLOUD = False
BOROMIR = True

# DEFINITIONS
IMAGE_DATA_PATH = r'C:/Users/20174534/Documents/BMT3/Q4/BEP/ISIC-2017_Training_Data/ISIC-2017_Training_Data/'
MODEL_PATH = ''
REPORT_PATH = '../reports/'
TRUTH_CSV = 'ISIC-2017_Training_Part3_GroundTruth.csv'
BATCH_SIZE = 20
TRUTH_PATH = '../data/'

if BOROMIR:
    IMAGE_DATA_PATH = r'C:/Users/20174534/Documents/BMT3/Q4/BEP/ISIC-2017_Training_Data/ISIC-2017_Training_Data/'

if TRIAL:
    STEPS_PER_EPOCH_MODEL_1 = 4
    EPOCHS_MODEL_1 = 2
    STEPS_PER_EPOCH_MODEL_2 = 4
    EPOCHS_MODEL_2 = 6
else:
    STEPS_PER_EPOCH_MODEL_1 = 100
    EPOCHS_MODEL_1 = 30
    STEPS_PER_EPOCH_MODEL_2 = 40
    EPOCHS_MODEL_2 = 60

# IMPORTS
import keras.backend.tensorflow_backend
from keras.layers import Input
from sklearn.metrics import roc_auc_score
from generate_data import generate_data_1
from get_data import get_data_1
from report_results import report_acc_and_loss, report_auc
import numpy as np
import ipykernel

def read_data(seed):
    global test_id, test_label_c, class_weights, train, validation
    global train_id, train_label_c, valid_id, valid_label_c, test_id, test_label_c, class_weights

    train_id, train_label_c, valid_id, valid_label_c, test_id, test_label_c, class_weights = get_data_1(TRUTH_PATH, TRUTH_CSV, seed, VERBOSE, SANITY_CHECK)
    train = generate_data_1(directory=IMAGE_DATA_PATH, augmentation=True, batchsize=BATCH_SIZE, file_list=train_id,
                          label_1=train_label_c)
    validation = generate_data_1(directory=IMAGE_DATA_PATH, augmentation=False, batchsize=BATCH_SIZE, file_list=valid_id,
                               label_1=valid_label_c)
def build_model():
    # instantiate the convolutional base
    img_height, img_width, img_channel = 384, 384, 3
    conv_base = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet',
                                               input_shape=(img_height, img_width, img_channel))
    # add a densely connected classifier on top of conv base
    model = keras.models.Sequential()
    model.add(conv_base)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    if VERBOSE:
        print('Trainable weights before freezing conv base = ' + str(len(model.trainable_weights)))
    # freeze the conv base
    conv_base.trainable = False
    if VERBOSE:
        print('Trainable weights after freezing conv base = ' + str(len(model.trainable_weights)))

    model.compile(loss='binary_crossentropy',
                  #optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9),
                  optimizer=keras.optimizers.RMSprop(lr=2e-5),
                  metrics=['acc'])
    return model

def fit_model(model):
    global history
    model.summary()
    history = model.fit_generator(
        train,
        steps_per_epoch=STEPS_PER_EPOCH_MODEL_1,
        epochs=EPOCHS_MODEL_1,
        validation_steps=50,
        validation_data=validation,
        class_weight=class_weights)

def predict_model(model):
    test = generate_data_1(directory=IMAGE_DATA_PATH, augmentation=False, batchsize=BATCH_SIZE, file_list=test_id,
                         label_1=test_label_c)
    predictions = model.predict_generator(test, steps=25)

    y_true = test_label_c
    delta_size = predictions.size - y_true.count()
    scores = np.resize(predictions, predictions.size - delta_size)
    auc = roc_auc_score(y_true, scores)
    return auc

aucs = []
if TRIAL:
    seeds = [1970, 1972]
else:
    seeds = [1970, 1972, 2008, 2019, 2020]
for seed in seeds:
    read_data(seed)
    model = build_model()
    fit_model(model)
    report_acc_and_loss(history, REPORT_PATH, seed)
    score = predict_model(model)
    aucs.append(score)
report_auc(aucs, REPORT_PATH)