import keras
from generate_data import generate_data_2, Generate_Alt_2
from get_data import get_data_2, annototation_type
from report_results import report_acc_and_loss, report_auc
from sklearn.metrics import roc_auc_score, plot_confusion_matrix
from scipy import optimize
import numpy as np
import os
from vis.visualization import visualize_cam
from vis.utils import utils
import matplotlib.pyplot as plt

# FLAGS
# Types are weighted or non_weighted
SANITY_CHECK = False
VERBOSE = True
WEIGHTED = False
VISUALISE_CAM = True

# Definitions

IMAGE_DATA_PATH = 'C:\\Users\\max\\stack\\TUE\\Sync_laptop\\data_bep\\isic-challenge-2017\\ISIC-2017_Training_Data\\'
MODEL_PATH = ''
REPORT_PATH = '../reports/'
WEIGHTS_PATH = '../weights/'
TRUTH_CSV = 'ISIC-2017_Training_Part3_GroundTruth.csv'
BATCH_SIZE = 20
TRUTH_PATH = '../data/'
GROUP_PATH = '../data/'
# Savename for aucs
SAVENAME = 'ensemble_weighted'
# Name of saved weights from experiments
ReportName = 'multitask'
Annotation_Type = annototation_type.asymmetry


def read_data(seed, annotation):
    global train_id, valid_id, test_id, train_label_c, valid_label_c, test_label_c, train_label_a
    global valid_label_a, test_label_a, train_mask, valid_mask, test_mask, class_weights
    global train, validation

    train_id, valid_id, test_id, train_label_c, valid_label_c, test_label_c, train_label_a, valid_label_a, test_label_a, train_mask, valid_mask, test_mask, class_weights = get_data_2(
        GROUP_PATH, TRUTH_PATH, TRUTH_CSV, seed, VERBOSE, SANITY_CHECK, annotation)


def load_model(seed, annotation, WeightsPath):
    # Load model from model type depending on savename for the model and filename end(asymmetry, border or color)
    folder_list = os.listdir(WeightsPath)
    filtered_reportname = [s for s in folder_list if ReportName in s]
    filtered_seed = [s for s in filtered_reportname if str(seed) in s]
    filtered_h5 = [s for s in filtered_seed if "h5" in s]
    filtered_json = [s for s in filtered_seed if "json" in s]
    try:
        if annotation == annotation.asymmetry:
            file_json = [s for s in filtered_json if s[-19:-10] == "asymmetry"][0]
            file_h5 = [s for s in filtered_h5 if s[-17:-8] == "asymmetry"][0]
            model_type = "asymmetry"
        if annotation == annotation.border:
            file_json = [s for s in filtered_json if s[-16:-10] == "border"][0]
            file_h5 = [s for s in filtered_h5 if s[-14:-8] == "border"][0]
            model_type = "border"
        if annotation == annotation.color:
            file_json = [s for s in filtered_json if s[-15:-10] == "color"][0]
            file_h5 = [s for s in filtered_h5 if s[-13:-8] == "color"][0]
            model_type = "color"
    except:
        print("Not all model files in folder or ReportName incorrect")
        raise
    json_file = open(WeightsPath + file_json, 'r')
    ModelJSON = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(ModelJSON)
    model.load_weights(WeightsPath + file_h5)
    return model, model_type


def ensemble_predictions(seeds):
    # predict from generator using loaded model file.
    gt_test_dict = {}
    predictions_dict_test = {}
    gt_val_dict = {}
    predictions_dict_val = {}
    for seed in seeds:
        ann_count = 0
        predictions_test = np.zeros([3, 250])
        predictions_val = np.zeros([3, 350])
        read_data(seed, Annotation_Type)
        for a_type in Annotation_Type:
            model, model_type = load_model(seed, a_type, WEIGHTS_PATH)
            test_gen = generate_data_2(directory=IMAGE_DATA_PATH,
                                       augmentation=False,
                                       batch_size=20,
                                       file_list=test_id,
                                       label_1=test_label_c,
                                       label_2=test_label_a,
                                       sample_weights=test_mask)
            val_gen = generate_data_2(directory=IMAGE_DATA_PATH,
                                      augmentation=False,
                                      batch_size=20,
                                      file_list=valid_id,
                                      label_1=valid_label_c,
                                      label_2=valid_label_a,
                                      sample_weights=valid_mask)

            model_pred_test = model.predict_generator(test_gen, 13)
            delta_size_test = model_pred_test[0].size - test_label_c.count()
            predictions_test[ann_count, :] = np.resize(model_pred_test[0], model_pred_test[0].size - delta_size_test)

            model_pred_val = model.predict_generator(val_gen, 18)  # Change amount of iterations
            delta_size_val = model_pred_val[0].size - valid_label_c.count()
            predictions_val[ann_count, :] = np.resize(model_pred_val[0], model_pred_val[0].size - delta_size_val)

            if VISUALISE_CAM:
                plot_gradCAM(model, test_gen, model_type, seed)

            plot_confusion_matrix(X=test_label_c, y_true=predictions_test[ann_count, :])

            ann_count += 1
        gt_test_dict[seed] = test_label_c
        gt_val_dict[seed] = valid_label_c
        predictions_dict_test[seed] = predictions_test
        predictions_dict_val[seed] = predictions_val
    return gt_test_dict, predictions_dict_test, gt_val_dict, predictions_dict_val


def auc_score_ensemble():
    # Ensemble model predictions
    for seed in seeds:
        gt_test = gt_test_dict[seed]
        predictions = predictions_dict_test[seed]
        predictions_mean = np.average(predictions, axis=0)
        auc = [roc_auc_score(gt_test, predictions_mean)]
        report_auc(auc, REPORT_PATH, seed, SAVENAME)


def loss_mse(weights, gt_val, predictions_val):
    predictions = np.average(predictions_val, weights=weights, axis=0)
    mse = keras.losses.MeanSquaredError()
    loss = mse(gt_val, predictions).numpy()
    return loss


def auc_score_ensemble_weigthed():
    # Ensemble model predictions using weight factor optimized on validation set.
    for seed in seeds:
        gt_val = gt_val_dict[seed]
        predictions_val = predictions_dict_val[seed]
        weights = np.array([1/3, 1/3, 1/3])
        # weigths_min = optimize.minimize(loss_mse,
        #                                 weights,
        #                                 args=(gt_val, predictions_val),
        #                                 method="Nelder-Mead",
        #                                 tol=1e-6,
        #                                 constraints=({'type': 'eq', 'fun': lambda w: 1-sum(w)}))
        weigths_min = optimize.differential_evolution(loss_mse,
                                                      bounds=[(0.0, 1.0) for _ in range(len(weights))],
                                                      args=(gt_val, predictions_val),
                                                      maxiter=1000,
                                                      tol=1e-7)
        gt_test = gt_test_dict[seed]
        predictions_test = predictions_dict_test[seed]
        predictions_weighted = np.average(predictions_test, weights=weigths_min.x, axis=0)
        print(weigths_min.x)
        auc = [roc_auc_score(gt_test, predictions_weighted)]
        report_auc(auc, REPORT_PATH, seed, SAVENAME)

def plot_gradCAM(model, test_gen, model_type, seed):
    # plot a Gradient-weighted Class Activation Mapping of the test images.
    last_layer = utils.find_layer_idx(model, "block5_conv3")
    batch = next(test_gen)
    number_in_batch = np.argwhere(batch[1]['out_class'] > 0.5)[3][0]
    # print(number_in_batch) # Debug
    image = batch[0][number_in_batch]
    GT = batch[1]['out_class'][number_in_batch]
    print(batch[1]['out_class'])
    prediction = model.predict(np.expand_dims(image, axis=0))[0][0][0]
    CAM_image = visualize_cam(model=model, layer_idx=last_layer, seed_input=image, filter_indices=None)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image)
    axes[1].imshow(image)
    axes[1].imshow(CAM_image, alpha=0.5, cmap='jet')
    plt.suptitle('Model type: ' + model_type + ', Prediction: ' + str(round(prediction, 2)) + ', Ground truth: ' + str(GT) + ', Seed: ' + str(seed))
    axes[1].set_title('Gradient-CAM of Test image')
    axes[0].set_title('Test image')
    plt.show()


seeds = [2008]#, 1972, 1970, 2019, 2020]
gt_test_dict, predictions_dict_test, gt_val_dict, predictions_dict_val = ensemble_predictions(seeds)
auc_score_ensemble()
