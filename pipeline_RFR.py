import math
import os
import random

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# description:
# section A - 5 functions for each step of the current pipeline
# section B - helper functions for each step (the related step is mentioned above the first helper function)
# section C - main

# before running:
# - take out of the comment the final lines in main
# - explanation on results and process in the WORD file :)


#####################
# section A:
#####################

def step_0(data_path):
    os.chdir(data_path)
    paths = os.listdir()
    dfs = upload_dfs(paths)
    return dfs


# step 1 - pre-processing:
def step_1_preprocessing(full_dfs_lst, full_dfs_dict):
    # handling metadata:
    tmp_dfs_lst = metadata_and_kegg_names_dfs(full_dfs_lst)
    metadata_df = tmp_dfs_lst[0]
    kegg_names_df = tmp_dfs_lst[1]

    # handling taxonomical data:
    similar_tax_data_dict = similar_data_dict(full_dfs_lst)  # dict[df_name: dict[feature: dict[val: count]]]
    df_names = similar_tax_data_dict.keys()  # ONLY for taxonomical dfs
    total_features_before_drop = 0
    omic_data_dict = {}

    for df_name in list(df_names):
        feature_to_val_counts_dict = similar_tax_data_dict[df_name]
        features = list(feature_to_val_counts_dict.keys())
        total_features_before_drop += len(features)
        feat_to_num_of_zero_vals_dict = sum_zero_vals(feature_to_val_counts_dict)
        df = full_dfs_dict[df_name]
        feat_to_Nan_count_dict = feature_to_count_Nan_dict(feature_to_val_counts_dict)
        func_output = drop_features_with_only_zero_or_Nan(df, feat_to_num_of_zero_vals_dict, feat_to_Nan_count_dict)
        new_df = func_output[0]
        dropped_features_lst = func_output[1]

        # updating the df variables after the drop:
        for feature in dropped_features_lst:
            features.remove(feature)
            if feat_to_num_of_zero_vals_dict.get(feature):
                feat_to_num_of_zero_vals_dict.pop(feature)
            elif feat_to_Nan_count_dict.get(feature):
                feat_to_Nan_count_dict.pop(feature)

        # at this point the new df still might contain Nan values
        # we'll replace them with 0 for now
        # NEEDS TO BE CHANGED LATER

        # list of features that contain Nan values:
        Nan_dict_keys = list(feat_to_Nan_count_dict.keys())
        for feat in Nan_dict_keys:
            if feat_to_Nan_count_dict.get(feat) == 0:
                feat_to_Nan_count_dict.pop(feat)
        lst_of_features_with_Nan_values = list(feat_to_Nan_count_dict.keys())

        # replacing each Nan value with 0:
        for feature in lst_of_features_with_Nan_values:
            new_df[feature] = new_df[feature].fillna(0)

        omic_data_dict[df_name] = new_df

    # at this point we have a dictionary from df names to data frames
    # each df is without features that were only 0

    return metadata_df, kegg_names_df, omic_data_dict


# step 2 - missing values imputation: (intially impute the average - improve later)
def step_2_missing_values_imputation(metadata_df):
    # metadata_without_Denmark = metadata_df.drop(metadata_df.index[metadata_df['CENTER'] == 'Denmark'], inplace=False)
    metadata_without_Nan = metadata_df.dropna()
    # the metadata without the Nan values will be used to compute the value to impute for the missing values
    features = list(metadata_without_Nan.columns)
    for feature in features:
        if feature == 'Gender':
            gender_female_percentage = metadata_without_Nan[feature].mean()
            values_to_fill = create_values_to_impute(gender_female_percentage, metadata_df[feature])
            metadata_df = fill_missing_values(values_to_fill, metadata_df, feature)
        elif feature == 'AGE':
            age_mean = metadata_without_Nan[feature].mean()
            metadata_df = fill_missing_values(age_mean, metadata_df, feature, ismean=True)
        elif feature == 'SMOKE':
            smokers_percentage = metadata_without_Nan[feature].mean()
            values_to_fill = create_values_to_impute(smokers_percentage, metadata_df[feature])
            metadata_df = fill_missing_values(values_to_fill, metadata_df, feature)

    return metadata_df


# step 3 - feature selection (intially with default selection to see if works - improve later)
def step_3_feature_selection(omic_data_dict, x=0.05):
    omic_df_names = list(omic_data_dict.keys())
    reduced_omic_data_dict = {}
    for name in omic_df_names:
        reduce_features_in_df(omic_data_dict, name, reduced_omic_data_dict, x)
    return reduced_omic_data_dict


# step 4 - choosing a model (initially do RF - improve later)
def step_4_model(metadata_df, reduced_omic_data_dict):
    # first, we want to merge all data to one dataframe:
    combined_data = combine_dfs(metadata_df, reduced_omic_data_dict)

    # now we have a data set with 109 features for x = 0.05:
    # 104 omic data features, 5 metadata features
    # those 104 are the ones with the highest variance

    # converting patient group to binary values:
    combined_data["PatientGroup"] = np.where(combined_data["PatientGroup"] == "8", 0, 1)

    # we would like to predict values for the feature 'Patient Group',
    # to do so we seperate it from the rest of the data:
    train_X = combined_data.drop(columns=['PatientGroup', 'CENTER'])
    train_Y = combined_data['PatientGroup']

    # randomly split the data
    train_x, test_x, train_y, test_y = train_test_split(train_X, train_Y, test_size=0.25, random_state=0)

    # create an object of the RandomForestRegressor
    model_RFR = RandomForestRegressor(max_depth=10)

    # fit the model with the training data
    model_RFR.fit(train_x, train_y)

    # predict the target on train and test data
    predict_train = model_RFR.predict(train_x)
    predict_test = model_RFR.predict(test_x)

    # Root Mean Squared Error on train and test data
    RMSE_on_train_data = mean_squared_error(train_y, predict_train) ** 0.5
    RMSE_on_test_data = mean_squared_error(test_y, predict_test) ** 0.5

    # creating a df with the results
    predict_train_df = create_results_df(train_y, predict_train)
    predict_test_df = create_results_df(test_y, predict_test)

    return RMSE_on_train_data, RMSE_on_test_data, predict_train_df, predict_test_df


#####################
# section B:
#####################

# step 0:
def upload_dfs(paths):
    relevant_paths = [path for path in paths if path.endswith(".txt")]
    dfs = []
    dfs_dict = {}
    for i in range(len(relevant_paths)):
        path = relevant_paths[i]
        name = path.split(".")[0]
        if path == "kegg_names.txt":
            dfs.append((pd.read_fwf(path, header=None, names=["SampleID", "pathway_name"]), path))
        elif path == "metadata.txt":
            dfs.append((pd.read_table(path, header=0), path))
        else:
            dfs.append((pd.read_csv(path, sep=' ', header=0), path))
        dfs[i][0].set_index("SampleID", inplace=True)
        dfs_dict[name] = dfs[i][0]
    return dfs, dfs_dict


# step 1:
def metadata_and_kegg_names_dfs(full_dfs_lst):
    metadata_df = None
    kegg_names_df = None
    for elm in full_dfs_lst:
        if "metadata" in elm[1]:
            metadata_df = elm[0]

        elif "kegg_names" in elm[1]:
            kegg_names_df = elm[0]

        if metadata_df is not None and kegg_names_df is not None:
            break
    return metadata_df, kegg_names_df


def similar_data_dict(dfs_lst):
    df_to_dict = {}  # dictionary with key --> df, entry --> new dictionary

    for elm in dfs_lst:
        df = elm[0]
        if "metadata" in elm[1] or "kegg_names" in elm[1]:
            continue
        else:
            col_to_val_dict = {}  # dictionary: key --> column, entry --> dictionary
            df_to_dict[elm[1].split(".")[0]] = col_to_val_dict
            for col in df:  # for each coulmn in dataframe we want to save a dictionary val to count
                val_to_count = {}  # dictionary: key --> value of sample, entry --> num of times it appears
                col_to_val_dict[df[col].name] = val_to_count
                val_to_count["Nan"] = 0
                for val in df[col].values:
                    if math.isnan(val):
                        val_to_count["Nan"] = val_to_count["Nan"] + 1
                    elif val not in val_to_count:
                        val_to_count[val] = 1
                    else:
                        curr_count = val_to_count[val]
                        val_to_count[val] = curr_count + 1

    return df_to_dict


def sum_zero_vals(feat_to_val_count_dict):
    keys = feat_to_val_count_dict.keys()
    features_to_num_of_zero_vals = {}
    for key in keys:
        val_counts_dict = feat_to_val_count_dict[key]
        vals = list(val_counts_dict.keys())
        if 0 in vals:
            features_to_num_of_zero_vals[key] = val_counts_dict.get(0)
    return features_to_num_of_zero_vals


def feature_to_count_Nan_dict(feat_to_val_count_dict):
    keys = feat_to_val_count_dict.keys()
    features_to_num_of_Nan_vals = {}
    for key in keys:
        val_counts_dict = feat_to_val_count_dict[key]
        vals = list(val_counts_dict.keys())
        if "Nan" in vals:
            features_to_num_of_Nan_vals[key] = val_counts_dict.get("Nan")
    return features_to_num_of_Nan_vals


def drop_features_with_only_zero_or_Nan(df, feat_to_num_zero, feat_to_Nans):
    lst_of_zero_features = list_of_features_with_single_val(feat_to_num_zero)
    lst_of_features_with_Nan = list_of_features_with_single_val(feat_to_Nans)
    drop_lst = lst_of_zero_features + lst_of_features_with_Nan
    filtered_features_df = df.drop(labels=drop_lst, axis=1)
    return filtered_features_df, drop_lst


def list_of_features_with_single_val(feat_to_sum_val_dict, total_samples=1058):
    features_single_val = []
    for feature in feat_to_sum_val_dict.keys():
        num_of_vals = feat_to_sum_val_dict.get(feature)
        if num_of_vals == total_samples:
            features_single_val.append(feature)
    return features_single_val


# step 2:
def create_values_to_impute(ratio, full_column):
    num_of_Nan = full_column.isna().sum()  # number of values we need to fill
    values_to_fill = []
    # we want catagorial values in the same ratio given
    num_of_ones = math.floor(num_of_Nan * ratio)
    num_of_zeros = num_of_Nan - num_of_ones
    for i in range(num_of_ones):
        values_to_fill.append(1.0)
    for j in range(num_of_zeros):
        values_to_fill.append(0.0)
    # we want the values in random order:
    random.shuffle(values_to_fill)
    return values_to_fill


def fill_missing_values(values_to_fill, full_df, feature, ismean=False):
    if ismean:
        full_df[feature].fillna(values_to_fill, inplace=True)
    else:  # the input is a list of values not the mean
        inds_to_fill = full_df.loc[pd.isna(full_df[feature]), :].index
        for i in range(len(inds_to_fill)):
            val = values_to_fill[i]
            full_df[feature].fillna(val, limit=1, inplace=True)
    return full_df


# step 3:
def reduce_features_in_df(omic_data_dict, df_name, reduced_omic_data_dict, x):
    curr_omic_df = omic_data_dict.get(df_name)
    var_curr_df = curr_omic_df.var()
    sorted_var_df = var_curr_df.sort_values()
    num_of_features = len(list(curr_omic_df.columns))
    num_of_features_to_keep = math.floor(num_of_features * x)
    num_of_features_to_keep += 2
    features = list(sorted_var_df.index)
    end = num_of_features - num_of_features_to_keep
    features_to_drop = features[0:end]
    reduced_df = curr_omic_df.drop(labels=features_to_drop, axis=1)
    reduced_omic_data_dict[df_name] = reduced_df


# step 4:
def combine_dfs(first_dfs, dfs_dict):
    dfs_names = list(dfs_dict.keys())
    curr_df = first_dfs
    for df_name in dfs_names:
        df = dfs_dict.get(df_name)
        curr_df = pd.concat([curr_df, df], ignore_index=False, axis=1)
    return curr_df


def create_results_df(y, predict_lst):
    results_df = pd.DataFrame(y)
    results_df["Predictions"] = predict_lst
    return results_df


# functions - section C
def success_rate(results_df, threshold=0.5):
    samples = results_df.index
    success_sum = 0
    samples_num = len(samples)
    for sample in samples:
        sample_is_sick = results_df.loc[sample, 'PatientGroup']
        pred_for_sample = results_df.loc[sample, 'Predictions']
        if sample_is_sick == 1 and pred_for_sample > threshold:
            success_sum += 1
        elif sample_is_sick == 0 and pred_for_sample < threshold:
            success_sum += 1
    suc_rate = success_sum / samples_num
    return suc_rate


#####################
# section C:
#####################

if __name__ == "__main__":
    # step 0 - reading raw data:
    dfs_tuple = step_0("data")
    dfs_lst_0 = dfs_tuple[0]
    dfs_dict_0 = dfs_tuple[1]

    # step 1 - preprocessing:
    # df_step_1 --> 2 dataframes and a dictionary of dataframes: metadata df, kegg names df, omic data dictionary
    # in metadata and kegg names Nan values were NOT handled
    # in omic data (metabolomic data) Nan was replaced with 0, columns with only the value 0 were removed
    dfs_step_1 = step_1_preprocessing(dfs_lst_0, dfs_dict_0)
    metadata = dfs_step_1[0]
    kegg_names = dfs_step_1[1]
    full_omic_data_dict = dfs_step_1[2]  # contains both metagenomic data (w/o functional data) and metabolomic data

    # step 2 - missing values imputation:
    # the imputation is only for the metadata
    # this is an intial imputation using the mean, needs to be improved using the taxonomical data
    metadata = step_2_missing_values_imputation(metadata)

    # step 3 - feature selection:
    reduced_omic_dfs_dict = step_3_feature_selection(full_omic_data_dict)

    # step 4 - choosing a model:
    RMSE_lst = step_4_model(metadata, reduced_omic_dfs_dict)
    RMSE_on_train = RMSE_lst[0]
    RMSE_on_test = RMSE_lst[1]
    prediction_train_df = RMSE_lst[2]
    prediction_test_df = RMSE_lst[3]

    # additional values for the predictions:
    success_rate_train = success_rate(prediction_train_df)
    success_rate_test = success_rate(prediction_test_df)

    #############
    # take out of comment:
    #############
    # """
    print("Root Mean Square Error - train data  ---->  " + str(RMSE_on_train))
    print("Root Mean Square Error - test data   ---->  " + str(RMSE_on_test))
    print("\n")
    print("The probability for a patient to be sick: (train data)")
    print(prediction_train_df)
    print("\n")
    print("The probability for a patient to be sick: (test data)")
    print(prediction_test_df)
    print("\n")
    print("The success rate for train data (threshold = 0.5)  ---->  " + str(success_rate_train))
    print("The success rate for test data (threshold = 0.5)  ---->  " + str(success_rate_test))

    # """

# added line for test
