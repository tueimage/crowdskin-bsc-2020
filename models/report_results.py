import pandas as pd


def report_acc_and_loss(history, report_path, seed, name=''):
    hist_df = pd.DataFrame(history.history)
    hist_csv_file = report_path + name + '_acc_loss_' + str(seed) + '.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)


def report_auc(aucs, report_path, seed='', name=''):
    aucs_df = pd.DataFrame(aucs)
    aucs_csv_file = report_path + name + '_aucs_' + str(seed) + '.csv'
    with open(aucs_csv_file, mode='w') as f:
        aucs_df.to_csv(f)
