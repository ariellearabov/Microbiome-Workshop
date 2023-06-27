import math

from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score

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
    # plt.title('Receiver Operating Characteristic')
    # plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.savefig('2nd_milestone_basic.png')
    # plt.show()
    return roc_auc


def precision_recall(true_classes, pred_classes, probs_array):
    # keep probabilities for the positive outcome only
    preds = probs_array[:, 1]
    precision, recall, _ = precision_recall_curve(true_classes, preds)
    auc_pr = auc(recall, precision)
    # f-score with emphasis on recall
    # f_score = np.mean(5*((precision*recall)/(4*precision + recall)))
    f1 = f1_score(true_classes, pred_classes, average='weighted')
    # plot the precision-recall curves
    no_skill = len(true_classes[true_classes == 1]) / len(true_classes)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='f1 score = %.3f' % f1)
    plt.plot(recall, precision, marker='.', label='AUC = %.3f' % auc_pr)
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()


def split_specific_patient_group(df, group_out="1"):
    """
    :param df: combined dataframes with "Patient Group" feature
    :param group_out: patient group to leave out of the train dataset
    :return: data split to train and test
    """
    # groups = ["1", "2a", "3", "4", "6", "8"]
    # for group in groups:
    #     group_size = len(df.loc[df["PatientGroup"] == group].index)
    #     print("PatientGroup - " + group + " ---> group size = " + str(group_size))

    df_with_only_group = df.loc[df["PatientGroup"] == group_out]
    df_with_only_healthy = df.loc[df["PatientGroup"] == "8"]

    healthy_ind = list(df_with_only_healthy.index)
    group_out_ind = list(df_with_only_group.index)

    healthy_size = len(healthy_ind)
    group_out_size = len(group_out_ind)
    # remainder_size = 1058 - healthy_size - group_out_size

    # frac = group_out_size / remainder_size
    frac = healthy_size / 1058

    # random.shuffle(healthy_ind)
    size_of_healthy_to_test = math.floor(frac * group_out_size)
    # healthy_ind_to_test = healthy_ind[0:size_of_healthy_to_test]
    # healthy_ind_to_test = healthy_ind[158:193]
    # healthy_ind_to_test = healthy_ind[158:193]
    healthy_ind_to_test = healthy_ind[193 - size_of_healthy_to_test:193]
    print(len(healthy_ind_to_test))
    test_ind_lst = healthy_ind_to_test + group_out_ind

    test_size = len(test_ind_lst)/1058
    print("test_size = " + str(test_size))

    df_train = df.drop(labels=test_ind_lst, axis=0)
    train_x = df_train.drop(columns=['PatientGroup', 'CENTER'])
    train_y = df_train['PatientGroup']

    drop = list(df_train.index)
    df_test = df.drop(labels=drop, axis=0)
    test_x = df_test.drop(columns=['PatientGroup', 'CENTER'])
    test_y = df_test['PatientGroup']

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
    pred_class_test = predictions[4]

    # plotting ROC-curve for train data:
    # ROC_AUC(true_class_train, probs_array_train)

    # plotting ROC-curve for test data:
    print(ROC_AUC(true_class_test, probs_array_test))
    precision_recall(true_class_test, pred_class_test, probs_array_test)

    # dfs_tuple = pip.step_0("data")
    # dfs_lst_0 = dfs_tuple[0]
    # dfs_dict_0 = dfs_tuple[1]
    # dfs_step_1 = pip.step_1_preprocessing(dfs_lst_0, dfs_dict_0)
    # metadata = dfs_step_1[0]
    # split_specific_patient_group(metadata)
