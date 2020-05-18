# IMPORTS
import keras
from keras.layers import Input
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import roc_auc_score
from generate_data import generate_data_2, Generate_Alt_2
from get_data import get_data_2, annototation_type
from report_results import report_acc_and_loss, report_auc
import numpy as np
import tensorflow as tf

# RUN FLAGS
TRIAL = False
SANITY_CHECK = False
DEBUG = False
VERBOSE = True
GOOGLE_CLOUD = False
BOROMIR = False
GENERATE_ALTERNATIVE = False

# DEFINITIONS
IMAGE_DATA_PATH = 'C:\\Users\\max\\stack\\TUE\\Sync_laptop\\data_bep\\isic-challenge-2017\\ISIC-2017_Training_Data\\'
MODEL_PATH = ''
REPORT_PATH = '../reports/'
WEIGHTS_PATH = '../weights/'
TRUTH_CSV = 'ISIC-2017_Training_Part3_GroundTruth.csv'
BATCH_SIZE = 5 #20
TRUTH_PATH = '../data/'
GROUP_PATH = '../data/'
ReportName = 'multitask'

if BOROMIR:
    IMAGE_DATA_PATH = '/data/ralf/19/'

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


def read_data(seed):
    global train_id, valid_id, test_id, train_label_c, valid_label_c, test_label_c, train_label_a
    global valid_label_a, test_label_a, train_mask, valid_mask, test_mask, class_weights
    global train, validation

    train_id, valid_id, test_id, train_label_c, valid_label_c, test_label_c, train_label_a, valid_label_a, test_label_a, train_mask, valid_mask, test_mask, class_weights = get_data_2(
        GROUP_PATH, TRUTH_PATH, TRUTH_CSV, seed, VERBOSE, SANITY_CHECK, annototation_type.border)
    if GENERATE_ALTERNATIVE:
        train = Generate_Alt_2(directory=IMAGE_DATA_PATH,
                               augmentation=True,
                               batch_size=BATCH_SIZE,
                               file_list=train_id,
                               label_1=train_label_c,
                               label_2=train_label_a,
                               sample_weights=train_mask,
                               class_weights=class_weights)
        validation = Generate_Alt_2(directory=IMAGE_DATA_PATH,
                                    augmentation=False,
                                    batch_size=BATCH_SIZE,
                                    file_list=valid_id,
                                    label_1=valid_label_c,
                                    label_2=valid_label_a,
                                    sample_weights=valid_mask,
                                    class_weights=class_weights)

    else:
        train = generate_data_2(directory=IMAGE_DATA_PATH,
                                augmentation=True,
                                batch_size=BATCH_SIZE,
                                file_list=train_id,
                                label_1=train_label_c,
                                label_2=train_label_a,
                                sample_weights=train_mask)
        validation = generate_data_2(directory=IMAGE_DATA_PATH,
                                     augmentation=False,
                                     batch_size=BATCH_SIZE,
                                     file_list=valid_id,
                                     label_1=valid_label_c,
                                     label_2=valid_label_a,
                                     sample_weights=valid_mask)


def mse(y_true, y_pred):
    mask = []
    for i in range(0, 10):
        if y_true[i] == 0:
            mask.append(0.0)
        else:
            mask.append(1.0)
    if all(value == 0 for value in mask):
        return 0.
    else:
        mask = np.array(mask)
        mask = K.cast(mask, K.floatx())
        score_array = K.square(y_true - y_pred)
        score_array *= mask
        score_array /= K.mean(K.cast(K.not_equal(mask, 0), K.floatx()))
        return K.mean(score_array)


def build_model():
    img_height, img_width, img_channel = 384, 384, 3
    conv_base = keras.applications.vgg16.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(img_height, img_width, img_channel))
    conv_base.trainable = False
    x = keras.layers.Flatten()(conv_base.output)
    x = keras.layers.Dense(256, activation='relu')(x)
    out_class = keras.layers.Dense(1, activation='sigmoid', name='out_class')(x)
    out_asymm = keras.layers.Dense(1, activation='linear', name='out_asymm')(x)
    model = keras.models.Model(conv_base.input, outputs=[out_class, out_asymm])
    model.compile(
        optimizer=keras.optimizers.RMSprop(lr=2e-5),
        loss={'out_class': 'binary_crossentropy', 'out_asymm': 'mse'},
        loss_weights={'out_class': 0.5, 'out_asymm': 0.5},
        metrics={'out_class': 'accuracy'})

    if VERBOSE:
        model.summary()
    return model


def fit_model(model):
    # remember to disable shuffling
    global history
    history = model.fit_generator(
        train,
        steps_per_epoch=STEPS_PER_EPOCH_MODEL_2,
        epochs=EPOCHS_MODEL_2,
        # class_weight={0: 1., 1: 3.},
        validation_data=validation,
        validation_steps=50,
        # callbacks=callbacks_list,
        shuffle=False)


def predict_model(model):
    if GENERATE_ALTERNATIVE:
        test = Generate_Alt_2(directory=IMAGE_DATA_PATH,
                              augmentation=False,
                              batch_size=BATCH_SIZE,
                              file_list=test_id,
                              label_1=test_label_c,
                              label_2=test_label_a,
                              sample_weights=test_mask,
                              class_weights=class_weights)
        predictions = model.predict_generator(test)
    else:
        test = generate_data_2(directory=IMAGE_DATA_PATH,
                               augmentation=False,
                               batch_size=BATCH_SIZE,
                               file_list=test_id,
                               label_1=test_label_c,
                               label_2=test_label_a,
                               sample_weights=test_mask)
        predictions = model.predict_generator(test, 25)
    delta_size = predictions[0].size - test_label_c.count()
    scores = np.resize(predictions[0], predictions[0].size - delta_size)
    auc = roc_auc_score(test_label_c, scores)
    return auc


def callbacks(weights_filepath):
    # setup callbacks for model fitting
    # save_location = weights_filepath + 'procedural_classification_' + str(seed) + '.hdf5'
    # checkpoint = ModelCheckpoint(save_location, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensorboard = TensorBoard('/content/logs/procedural_classification_' + str(seed), profile_batch='10,20')
    callbacks_list_out = [tensorboard]
    return callbacks_list_out


def save_model():
    model_json = model.to_json()
    path = WEIGHTS_PATH + ReportName + '_' + str(seed) + '.json'
    with open(path, 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(WEIGHTS_PATH + ReportName + '_' + str(seed) + '.h5')


aucs = []
if TRIAL:
    seeds = [1970, 1972]
else:
    seeds = [1970, 1972, 2008, 2019, 2020]
for seed in seeds:
    read_data(seed)
    model = build_model()
    callbacks_list = callbacks(WEIGHTS_PATH)
    fit_model(model)
    # save_model()
    report_acc_and_loss(history, REPORT_PATH, seed, ReportName)
    score = predict_model(model)
    aucs.append(score)
report_auc(aucs, REPORT_PATH, seeds[0], ReportName)
