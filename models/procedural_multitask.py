# IMPORTS
import keras.backend.tensorflow_backend
import numpy as np
from sklearn.metrics import roc_auc_score
from generate_data import generate_data_2
from get_data import get_data_2
from report_results import report_acc_and_loss, report_auc
from get_data import annototation_type

# RUN FLAGS
TRIAL = False
SANITY_CHECK = True
DEBUG = False
VERBOSE = True
GOOGLE_CLOUD = False
BOROMIR = True

# DEFINITIONS
IMAGE_DATA_PATH = '/data/CrowdSkin/ekcontar/dat/'
MODEL_PATH = ''
REPORT_PATH = '../reports/'
WEIGHTS_PATH = '../weights/'
TRUTH_CSV = 'ISIC-2017_Training_Part3_GroundTruth.csv'
BATCH_SIZE = 20
TRUTH_PATH = '../data/'
GROUP_PATH = '../data/'

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
        GROUP_PATH, TRUTH_PATH, TRUTH_CSV, seed, VERBOSE, SANITY_CHECK, annototation_type.asymmetry)

    train = generate_data_2(directory=IMAGE_DATA_PATH,
                            augmentation=True,
                            batch_size=BATCH_SIZE,
                            file_list=train_id,
                            label_1=train_label_c,
                            label_2=train_label_a,
                            sample_weights = train_mask)
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
    global history
    history=model.fit_generator(
        train,
        steps_per_epoch= STEPS_PER_EPOCH_MODEL_1,
        epochs=EPOCHS_MODEL_1,
        class_weight={0:1.,1:3.},
        validation_data=validation,
        validation_steps=50)

def predict_model(model):
    test = generate_data_2(directory=IMAGE_DATA_PATH,
                           augmentation=False,
                           batch_size=BATCH_SIZE,
                           file_list=test_id,
                           label_1=test_label_c,
                           label_2=test_label_a,
                           sample_weights = test_mask)
    predictions = model.predict_generator(test, 25)
    delta_size = predictions[0].size - test_label_c.count()
    scores=np.resize(predictions[0], predictions[0].size - delta_size)
    auc = roc_auc_score(test_label_c, scores)
    return auc

def save_model(model, seed):
    model_json = model.to_json()
    with open(WEIGHTS_PATH + 'model' + str(seed) + '.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(WEIGHTS_PATH + 'model' +  str(seed) + '.h5')

aucs = []
if TRIAL:
    seeds = [1970, 1972]
else:
    seeds = [1970, 1972, 2008, 2019, 2020]
    #seeds = [2008]
for seed in seeds:
    read_data(seed)
    model = build_model()
    fit_model(model)
    #save_model(model, seed)
    report_acc_and_loss(history, REPORT_PATH, seed)
    score = predict_model(model)
    aucs.append(score)
report_auc(aucs, REPORT_PATH)