import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

#path to reports
logpath = 'C:\\Users\\max\\stack\\TUE\\Sync_laptop\\BEP\\crowdskin-bsc-2020\\reports'
names_of_runs = ['procedural']

os.chdir(logpath)
list_of_files = os.listdir()

def aucs_df():
    aucs = pd.DataFrame()
    for run_name in names_of_runs:
        cur_filenames = [i for i in list_of_files if run_name+'_aucs' in i]
        for file_name in cur_filenames:
            auc = pd.read_csv(file_name)
            row_number = file_name[-8:-4]
            aucs.loc[row_number, run_name] = auc.iloc[0,1]
    return aucs

def loss_acc(run_name):
    acc = pd.DataFrame()
    loss = pd.DataFrame()
    cur_filenames = [i for i in list_of_files if run_name+'_acc_loss' in i]
    for file_name in cur_filenames:
        df_acc_loss = pd.read_csv(file_name)
        row_number = file_name[-8:-4]
        acc[row_number] = df_acc_loss['acc']
        acc['val_'+row_number] = df_acc_loss['val_acc']
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

def plot_aucs():
    aucs = aucs_df()
    plt.figure()
    plt.boxplot(x=aucs['procedural'])
    plt.ylabel('auc')
    plt.title('auc for all models')
    plt.show()

acc, loss = loss_acc('procedural')
plot_acc_loss(acc, loss)
plot_aucs()