import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

#path to reports
logpath = 'C:\\Users\\max\\stack\\TUE\\Sync_laptop\\BEP\\crowdskin-bsc-2020\\reports'
names_of_runs = ['procedural_normal',
                 # "multitask_asymmetry",
                 # "multitask_border",
                 # 'multitask_color',
                 # "ensemble_vgg16",
                 # "multitask_efficientnet_asymmetry",
                 # "multitask_efficientnet_border",
                 # "multitask_efficientnet_color",
                 # "ensemble_efficientnetb1",
                 # "multitask_inception_asymmetry",
                 # "multitask_inception_border",
                 # "multitask_inception_color",
                 # "ensemble_inceptionv3",
                 # "multitask_resnet_asymmetry",
                 # "multitask_resnet_border",
                 # "multitask_resnet_color",
                 # "ensemble_resnet50v2",
                 "ensemble_asymmetry",
                 "ensemble_border",
                 "ensemble_color",
                 'ensemble_multi_model']
os.chdir(logpath)
list_of_files = os.listdir()

def aucs_df():
    aucs = pd.DataFrame()
    for run_name in names_of_runs:
        cur_filenames = [i for i in list_of_files if run_name+'_aucs' in i]
        for file_name in cur_filenames:
            auc = pd.read_csv(file_name)
            row_number = file_name[-8:-4]
            aucs.loc[row_number, run_name] = auc.iloc[0, 1]
    return aucs

def loss_acc(run_name):
    acc = pd.DataFrame()
    loss = pd.DataFrame()
    if 'procedural' in run_name:
        cur_filenames = [i for i in list_of_files if run_name+'_acc_loss' in i]
        for file_name in cur_filenames:
            df_acc_loss = pd.read_csv(file_name)
            row_number = file_name[-8:-4]
            acc[row_number] = df_acc_loss['acc']
            acc['val_'+row_number] = df_acc_loss['val_acc']
            loss[row_number] = df_acc_loss['loss']
            loss['val_'+row_number] = df_acc_loss['val_loss']
    else:
        cur_filenames = [i for i in list_of_files if run_name+'_acc_loss' in i]
        for file_name in cur_filenames:
            df_acc_loss = pd.read_csv(file_name)
            row_number = file_name[-8:-4]
            acc[row_number] = df_acc_loss['out_class_accuracy']
            acc['val_'+row_number] = df_acc_loss['val_out_class_accuracy']
            loss[row_number] = df_acc_loss['loss']
            loss['val_'+row_number] = df_acc_loss['val_loss']

    return acc, loss

def plot_acc_loss(acc, loss):
    plt.figure()
    plt.plot(acc[acc.columns[::2]], color='blue')
    plt.plot(acc[acc.columns[1::2]], color='orange')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()
    plt.figure()
    plt.plot(loss[loss.columns[::2]], color='blue')
    plt.plot(loss[loss.columns[1::2]], color='orange')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

def plot_aucs(aucs):
    #simple function for plotting aucs
    plt.figure()
    aucs.boxplot(rot=-0)
    plt.xticks(np.arange(1, 7), ['Baseline', 'VGG16', 'EfficientNetB1', 'InceptionV3', 'ResNet50V2', 'Ensemble'], fontsize=8)
    # plt.xticks(np.arange(1, 6), ['Baseline', 'Asymmetry', 'Border', 'Color', 'Ensemble'],
    #            fontsize=8)
    plt.ylabel('AUC')
    plt.title('AUC of feature ensemble')
    # plt.savefig('C:\\Users\\max\\stack\\TUE\\Sync_laptop\\BEP\\crowdskin-bsc-2020\\Visualisation and misc\\feature_ensemble.svg')
    plt.show()

def print_mean(aucs):
    # print mean of selected aucs
    for run_name in names_of_runs:
        print(run_name+':', end='')
        print((34-len(run_name)) * ' ', end='')
        print(str(format(round(aucs[run_name].mean(), 3), '.3f')) + '±' + str(format(round(aucs[run_name].std(), 3), '.3f')))

# acc, loss = loss_acc(names_of_runs[1])
# plot_acc_loss(acc, loss)
aucs = aucs_df()
plot_aucs(aucs)
print_mean(aucs)
