import math
import random

from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np
import pipeline as pip


##########################################################################################################
# test file - to use on different pipelines change the pipeline on the import line: import pipeline as pip
##########################################################################################################

def run_model(specific_split=False, split_func=None):
    if not specific_split:
        return pip.main()
    else:
        return pip.main(split_func)


def ROC_AUC(true_classes, probs_array):
    """
    function plots the ROC-curve with the AUC value, to save the image - take plt.savefig() out of comment.
    :param true_classes: array of the true classifications for the samples
    :param probs_array: 2-dimesional array with the probabilities for class == 0, class == 1
    :return: AUC value
    """
    preds = probs_array[:, 1]
    # using the built-in function to calculate the ROC-curve:
    fpr, tpr, threshold = metrics.roc_curve(true_classes, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # plt.savefig('ROC_curve.png')
    plt.show()
    return roc_auc


def split_specific_patient_group(df, group_out="1", test_size=0.3):
    """
    :param df: combined dataframes with "Patient Group" feature
    :param group_out: patient group to leave out of the train dataset
    :param test_size: fraction of the samples to leave out for test set
    :return: data split to train and test
    """

    total_samples = df.shape[0]
    total_for_test = math.floor(test_size * total_samples)

    df_with_only_group = df.loc[df["PatientGroup"] == group_out]
    df_healthy = df.loc[df["PatientGroup"] == "8"]

    group_size = df_with_only_group.shape[0]
    remainder_to_test = total_for_test - group_size

    healthy_ind_lst = list(df_healthy.index)
    random.shuffle(healthy_ind_lst)

    test_ind_lst = healthy_ind_lst[0:remainder_to_test] + list(df_with_only_group.index)

    # splitting df to train and test:
    train_df = df.drop(labels=test_ind_lst, axis=0)
    test_df = df.loc[test_ind_lst]
    # pd.set_option('display.max_rows', None)

    test_y = test_df['PatientGroup']
    test_x = test_df.drop(columns=['PatientGroup', 'CENTER'])

    train_y = train_df['PatientGroup']
    train_x = train_df.drop(columns=['PatientGroup', 'CENTER'])

    # converting patient group to binary values:
    test_y = np.where(test_y == "8", 0, 1)
    train_y = np.where(train_y == "8", 0, 1)

    return train_x, test_x, train_y, test_y


if __name__ == "__main__":
    # predictions = run_model()
    predictions = run_model(specific_split=True, split_func=split_specific_patient_group)

    probs_array_train = predictions[0]
    probs_array_test = predictions[1]
    true_class_train = predictions[2]
    true_class_test = predictions[3]

    # plotting ROC-curve for train data:
    ROC_AUC(true_class_train, probs_array_train)

    # plotting ROC-curve for test data:
    ROC_AUC(true_class_test, probs_array_test)

    # dfs_tuple = pip.step_0("data")
    # dfs_lst_0 = dfs_tuple[0]
    # dfs_dict_0 = dfs_tuple[1]
    # dfs_step_1 = pip.step_1_preprocessing(dfs_lst_0, dfs_dict_0)
    # metadata = dfs_step_1[0]
    # split_specific_patient_group(metadata)
