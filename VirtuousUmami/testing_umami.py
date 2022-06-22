"""
testing_umami.py

Script for running the model created by the previous step on the test data.

Input:
        testSet (String): the test set with only the features selected by the modeller included

Output:
        result_labels.txt: the predicted labels and proababilities by the model

Example run:
    For two class:
        python3 testing_umami.py testing_dataset.txt
"""
import sys
import os
import time
import datetime
import logging
import numpy as np
import csv
import shutil
import pandas as pd


from knnimpute import (
    knn_impute_optimistic
)
import copy
import joblib
from joblib import Parallel, delayed


def initLogging():
    """
    Purpose: sets the logging configurations and initiates logging
    """
    todaystr = datetime.date.today().strftime("%Y%m%d")
    logging.basicConfig(filename="{}/umami_run_report_{}.log".format(os.getcwd(), todaystr),
                        level=logging.DEBUG, format='%(asctime)s\t %(levelname)s\t%(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')


def select_predictor(dataset_initial, features, models_zip, training_labels_filename, output_folder,
                     thread_num, user='unknown', jobid=0, pid=0):
    """
    Selects which predictor to run according to the values of the selection_flag.
    :param dataset_initial: (2d array): array with the dataset and its feature values
    :param features: (list): datasets features names list
    :param models_zip: (String) path to zip file with model files and supplementary files
    :param training_labels_filename: (file) original dataset labels used for training these models
    :param output_folder: path for runtime and produced files
    :param thread_num: (integer) number of threads used for parallel calculations
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    """
    result = []

    result, modelfiles = parse_models_folder(models_zip, output_folder + 'models/', user, jobid, pid)
    if not result:
        return [0, modelfiles]

    result = predictor(dataset_initial, features, modelfiles[0], modelfiles[2], modelfiles[1], thread_num,
                                    user, jobid, pid)
    if result[0]:
        create_labels_file(result[1], training_labels_filename, output_folder, user, jobid, pid)
        os.remove(output_folder + "preprocessed_data.txt")
        os.remove(output_folder + "preprocessed_dataset_0.tsv")
        os.remove(output_folder + "info.txt")
        shutil.rmtree(output_folder + "models")
        return [1, "Successful completion!"]
    return [0, result[1]]


def parse_models_folder(zip_folder, out_folder, user='unknown', jobid=0, pid=0):
    """
    Decompress the model folder, and create the appropriate files and variables
    :param zip_folder:
    :param out_folder:
    :param user:
    :param jobid:
    :param pid:
    :return:
    """
    model_files = [[], '', '']
    try:
        shutil.unpack_archive(zip_folder, out_folder, 'zip')
        models = os.listdir(out_folder)
    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tError, could not unpack archived model files.".format(pid, jobid,
                                                                                                          user))
        return 0, "Error, could not unpack archived model files."
    for file in models:
        if file.endswith(".pkl") or file.endswith(".pkl.z"):
            model_files[0].append(out_folder + file)
        elif file.endswith("classifierChain.csv"):
            model_files[1] = out_folder + file
        else:
            model_files[2] = out_folder + file
    model_files[0] = sorted(model_files[0])
    return True, model_files


def create_labels_file(predictions, training_labels_file, output_folder, user, jobid, pid):
    """
    Write the predicted labels in output file, if they were originally alphanumeric transform them
    :param predictions: list of predictions
    :param training_labels_file: training labels file
    :param output_folder: outputfolder for the created labels file
    :param user: thos job's user
    :param jobid: this job's ID
    :param pid:this job's PID
    :return:
    """
    predictions = transform_labels_to_alpha_twoclass(predictions, parse_training_labels(training_labels_file))
    try:
        with open(output_folder + "result_labels.txt", "w") as result_labels_fid:
            i = 0
            for label in predictions[0]:
                if str(label) == 'non-umami':
                    result_labels_fid.write(str(label) + "\t" + str(1 - (predictions[1][i])) + "\n")
                else:
                    result_labels_fid.write(str(label) + "\t" + str(predictions[1][i]) + "\n")
                i += 1
    except IOError:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tError in producing predicted labels file.".format(pid, jobid, user))

def transform_labels_to_alpha_twoclass(labels, unique_labels):
    """
    Transforms numeric labels back to alphanumeric according to given unique_labels one to one mapping.
    :param labels: the input labels
    :param unique_labels: the list with the two unique labels
    :return: new_labels: the decoded labels with alphanumeric names
    """

    new_labels = []
    probabilities = []
    for x in labels:
        if x[0] == 0:
            new_labels.append(unique_labels[0])
            probabilities.append(x[1])
        elif x[0] == 1:
            new_labels.append(unique_labels[1])
            probabilities.append(x[1])

    # logging.info("Labels transformed to alphanumeric successfully.")
    return [new_labels, probabilities]

def parse_training_labels(labels_filename):
    """
    Find original training Unique Labels Set
    :param labels_filename: training labels file
    :return: set of unique labels
    """

    delim_labels = find_delimiter(labels_filename)

    # Parse input files
    labels = list()
    with open(labels_filename, "r") as labels_fid:
        for line in csv.reader(labels_fid, delimiter=delim_labels):
            for word in line:
                labels.append(word.strip())
    unique_labels = sorted(list(set(labels)))

    return unique_labels


def preprocess_testing_dataset(test_filename, maximums_filename, minimums_filename,
                               training_features_file, missing_imputation_method, normalization_method,
                               has_feature_headers, has_sample_headers, variables_for_normalization_string,
                               output_folder, data_been_preprocessed_flag, user='unknown', jobid=0, pid=0):
    """
    Perform the same preprocessing steps as the training dataset, or based to the selected specifications to the
     testing dataset
    :param test_filename: Test set file
    :param maximums_filename: maximums filename, used for arithmetic normalization
    :param minimums_filename:  minimums filename, used for arithmetic normalization
    :param training_features_file: selected training features file
    :param missing_imputation_method:  Missing imputation to use, 1 for Average 2 for KNN
    :param normalization_method: Normalization method to use 1 for Arithmetic and 2 for logarithmic
    :param has_feature_headers: if the input dataset has features headers
    :param has_sample_headers: if the input dataset has sample headers
    :param variables_for_normalization_string: list of features to normalize seperately
    :param output_folder: output folder for created files
    :param data_been_preprocessed_flag: If the testing dataset needs preprocessing
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's PID
    :return: True, preprocessed dataset and feature list if successful, ot False and the error message
    """

    # read max-min csvs as list, this is full list of omics before duplicate averaging
    if maximums_filename and minimums_filename and normalization_method == 1 and data_been_preprocessed_flag == 0:
        maximums = read_csv_to_float(maximums_filename, user, jobid, pid)
        minimums = read_csv_to_float(minimums_filename, user, jobid, pid)
    else:
        maximums = []
        minimums = []

    end_code, preprocessed_dataset, testing_features = preprocess_data(
        test_filename, maximums, minimums, training_features_file, missing_imputation_method,
        normalization_method, output_folder, data_been_preprocessed_flag, variables_for_normalization_string,
        has_feature_headers, has_sample_headers, user=user, jobid=jobid, pid=pid)
    if end_code:
        # return the preprocessed dataset and features
        return True, preprocessed_dataset, testing_features
    else:
        # return that an error occurred and a message
        return False, preprocessed_dataset, ''


def run_all(testset_filename, maximums_filename, minimums_filename,
            features_filename, missing_imputation_method, normalization_method, model_filename, selection_flag,
            data_been_preprocessed_flag, variables_for_normalization_string, has_features_header,
            has_samples_header, training_labels_filename, length_of_features_from_training_filename, output_folder, thread_num=2,
            user='unknown', jobid=0, pid=0):
    """
    Selects which predictor to run according to the values of the selection_flag.

    Args:
        testset_filename (string): the testset filename
        maximums_filename: (String) the filename with the maximum values of each feature
        minimums_filename: (String) the filename with the minimum values of each feature
        features_filename: (String) the filename with the indexes of the selected features extracted from the training
         step
        normalization_method: (Integer) 1 for arithmetic, 2 for logarithmic
        missing_imputation_method: (Integer) 1 for average imputation, 2 for KNN imputation
        model_filename: (String) the model filename
        selection_flag (integer): 0 for multi-class, 1 for regression, 2 for two-class
        data_been_preprocessed_flag: (integer) 0 if data haven't been preprocessed and 1 if they have
        variables_for_normalization_string: (String) the selected variables for normalization as a string with comma or
         newline separated strings, eg. "ABC,DEF,GHI"
        has_features_header: 1 if it has features header, 0 if it doesn't have
        has_samples_header: 1 if it has samples header, 0 if it doesn't have
        training_labels_filename (String): the filename of the training labels
        length_of_features_from_training_filename: (String) the filename with the length of features of the training set
         (from step 04)
        output_folder (String): the output folder
        thread_num: number of available threads dor parallel processes
        user (String): this job's username
        jobid (Integer): this job's ID in biomarkers_jobs table
        pid (Integer): this job's PID
    """


    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    error, testset, features = preprocess_testing_dataset(
        testset_filename, maximums_filename, minimums_filename, features_filename,
        missing_imputation_method, normalization_method, has_features_header, has_samples_header,
        variables_for_normalization_string, output_folder, data_been_preprocessed_flag, user, jobid, pid)

    if error:
        try:
            result = select_predictor(testset, features, model_filename, training_labels_filename,
                                      output_folder, thread_num, user, jobid, pid)
        except ValueError:
            logging.exception("PID:{}\tJOB:{}\tUSER:{}\tException while predicting.".format(pid, jobid, user))
            return [0, "Exception while predicting, please check if the dataset is correct for this model"]
    else:
        logging.error("PID:{}\tJOB:{}\tUSER:{}\tException while running preprocessing.".format(pid, jobid, user))
        return [0, "Exception while running parsing and preprocessing: {}".format(testset), '']

    return result


# ######## prediction class ###############################################

class PredictTwoclass:
    def __init__(self, inputx, model_list, features_list, multilabel, dict_chain, predict_data_initial,
                 feature_namelist, predict_proteins, thread_num):
        # variables for these parameters
        self.model_list = model_list
        self.inputx = inputx
        self.features_list = features_list  # model specific list of features selected
        # sequence of feature names written in front1 feature final is alphabetic
        # whereas features for dataset is not alphabetic. SO need to do name matching to read right values
        self.feature_namelist = feature_namelist
        self.predict_proteins = predict_proteins
        self.multilabel = multilabel
        self.dict_chain = dict_chain  # this is list of list holding classifier chain for each model
        self.missingFeatThreshold = 0.5
        self.predict_data_initial = np.transpose(np.array([np.array(x) for x in predict_data_initial]))
        if int(thread_num) > 2:
            self.threads = int(thread_num) - 2
        else:
            self.threads = 1

    def predict_fun_parallel(self):
        # ##### Load the Finalized Model from disk using joblib

        row_count = self.inputx.shape[0]

        # counter to pick 1st patient predicted as patients may not be predicted at all if missing
        # features exceeds threshold

        # iterate on each patient, check for missing % of features in original data and choose the models
        # for patient in range(5,6): # for testing
        mean_class = \
            Parallel(n_jobs=self.threads, verbose=False)(delayed(self.predict_fun_thread)(patient)
                                                      for patient in range(row_count))

        # mean_class = np.hstack((mean_class, np.array(predicted_patient).reshape(-1, 1)))

        return mean_class

    def predict_fun_thread(self, patient):

        # get the value for 1st key and choose this as seq to align other outputs i.e. y
        y_chain = self.dict_chain[0]
        pred_array = []  # initialise prediction class to empty list
        counter = 0  # counter for model count
        for mdl in copy.deepcopy(self.model_list):  # iterate over map object

            colindices = []  # feature indices selected for after normalisation and impute predict data
            clf = joblib.load(mdl)

            # get features and map inputX to shape of model i.e. features uses
            # X is of shape [n_samples, n_features]; counter maps to model list counter
            for feature in range(len(self.features_list[0])):  # iterate on features
                if self.features_list[counter][feature] == 1:  # append only those feature that are 1 i.e. selected
                    colindices.append(feature)  # get the indices of features selected

            # read specific patient i.e. observation record for the feature selected
            new_x = self.inputx[[patient], :]  # get the row
            new_x = new_x[:, colindices]  # get the columns

            # get unprocessed X for selected features to do missing feature check
            unprocessed_x = self.predict_data_initial[[patient], :]  # get rows
            unprocessed_x = unprocessed_x[:, colindices]  # get teh columns
            # nan signifies missing value
            missing_count = len(unprocessed_x[np.where(unprocessed_x == float("nan"))])
            total_feature = len(colindices)
            missing_pct = float(missing_count / total_feature)

            counter = counter + 1
            if missing_pct >= self.missingFeatThreshold:
                continue  # go to next model for prediction

            else:

                if self.multilabel:  # multi label problem
                    for i, c in enumerate(clf):  # guided by classifier chain, picks the sequence of Ys
                        if i == 0:
                            try:
                                y_pred = (c.predict(new_x)).reshape(-1, 1)
                            except ValueError:
                                y_pred = [np.array([0.])]
                        else:
                            # add the prior y or Ys predicted to X
                            input_stacked = np.hstack((new_x, y_pred))
                            try:
                                new_y = c.predict(input_stacked)  # predict y using the model trained for X and Y
                            except ValueError:
                                new_y = np.array([0.])
                            y_pred = np.hstack((y_pred, new_y.reshape(-1, 1)))

                    if counter == 1:  # 1st classifier chain is chosen as base, so start from 2nd
                        # list of ndarrays containing prediction per patient for each model
                        pred_array.append(y_pred)
                    else:
                        # swap y_pred guided by y_chain
                        new_list = y_pred.tolist()
                        shuffle_y = []
                        new_chain = self.dict_chain[counter - 1]
                        shuffle_y.append([])
                        for j in range(len(y_chain)):
                            for k in range(len(new_chain)):
                                if y_chain[j] == new_chain[k]:  # compare positions
                                    shuffle_y[0].append(new_list[0][k])
                        # list of ndarrays containing prediction per patient for each model
                        pred_array.append(shuffle_y)
                else:
                    # single label problem
                    p2 = clf.predict_proba(new_x)[:, 1]  # X is single sample
                    p2 = np.reshape(p2, (p2.shape[0]))
                    pred_array.append(p2)

        # ## get majority vote or mean class for each patient across multilabel (change this calculation for each type
        # of prediction)
        if len(pred_array) == 0:  # i.e. pred_array = [] if all models skipped and no prediction for the patient
            return None, patient
        else:
            patient_mean_1 = np.mean(pred_array, axis=0)
            if patient_mean_1 >= 0.5:
                patient_mean = 1.0  # so that false negatives are minimised
            else:
                patient_mean = 0.0

        mean_class = patient_mean

        return [mean_class, patient_mean_1[0]]


# function to read dictionary
def read_dict(file_name):
    chain_dict = []
    with open(file_name, 'r') as f:
        readline = csv.reader(f)
        for i, row in enumerate(readline):  # read each row to create list of lists
            chain_dict.append([])
            chain_dict[i].extend(row)

    return chain_dict


# function to read csv cells as float
def read_csv_to_float(file_name, user='unknown', jobid=0, pid=0):
    data = []
    with open(file_name, 'r') as f:
        read = csv.reader(f)
        for row in read:  # convert yo float
            for cell in row:
                try:
                    data.append(float(cell))
                except ValueError:
                    logging.exception(
                        "PID:{}\tJOB:{}\tUSER:{}\tError in csv reader {} not convertable to float."
                            .format(pid, jobid, user, cell))
    return data


# function to check duplicate features for original prediction data without normalisation and impute
# adapted from script written for Missense SNPs ensemble_multiclassifier.py by Konstantinos Theofilatos
def average_dup_feat_predict(dataset_initial, markers):
    dataset = {}  # initialise dictionary to hold features as key and sum of its multiple observations as list value
    dict_of_occurences = {}  # initialise dictionary to hold feature as key and its occurences as value
    num_of_elements = 0  # initialise counter for features iterated
    column_count = len(dataset_initial[0])  # counter for iterating on a feature across observations
    row_count = len(dataset_initial)  # row count is features count

    for i in range(row_count):
        if markers[i] not in dataset:  # if feature not already present in dictionary then add
            dict_of_occurences[markers[i]] = 1  # counter of occurence of feature set to 1
            dataset[markers[i]] = []  # initialise list to hold value against specific feature key
            for j in range(column_count):
                # exclude null values for average calculation
                if dataset_initial[i][j] != float("nan") and dataset_initial[i][j] != '':
                    dataset[markers[i]].append(float(dataset_initial[i][j]))  # append columns to feature key
                else:
                    dataset[markers[i]].append(float("nan"))  # append float("nan") (and not zero) for missing values

        else:
            dict_of_occurences[markers[i]] += 1  # increment the counter of occurence

            # if feature key already exists then do column specific addition
            for j in range(column_count):
                # exclude null values for average calculation
                if dataset_initial[i][j] != float("nan") and dataset_initial[i][j] != '':
                    dataset[markers[i]][j] = dataset[markers[i]][j] + float(dataset_initial[i][j])

            num_of_elements += 1  # increment counter for features iterated

    # calculate average for each feature key
    for key in dataset:  # iterate over keys
        for j in range(len(dataset[key])):
            dataset[key][j] = dataset[key][j] / dict_of_occurences[key]

    data = []  # initialise list to hold average value
    markers = []  # initialise list to hold feature names
    num_of_markers = 0

    # segregate average data and features
    for key, vals in dataset.items():
        data.append([])
        markers.append(key)
        for i in range(len(vals)):
            data[num_of_markers].append(vals[i])
        num_of_markers += 1

    return [data, markers]  # return average data and features


def predictor(dataset_initial, features, model_list, model_feature_file, classification_chain, thread_num,
              user, jobid, pid):
    """
    Predict TwoClass labels for a given dataset using the provided model list
    :param dataset_initial: input dataset for prediction
    :param features: input dataset's features names
    :param model_list: list of model files used for prediction
    :param model_feature_file: file with the selected features per model in the list
    :param classification_chain: file with the classification chain
    :param thread_num: number of available threads, used for parallel processes
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :return: the predicted data
    """

    # ##### Make Predictions on New Data
    # get original prediction data with feature matching and dup avg for missing val threshold check
    # don't change missing val i.e. -1000 to zero
    try:
        data_org_pred, proteins = average_dup_feat_predict(copy.deepcopy(dataset_initial), features)
        data_tran = np.transpose(dataset_initial)
        features_df = pd.read_csv(model_feature_file)
        # get the classifier chain
        dict_chain = read_dict(classification_chain)
        prediction = PredictTwoclass(data_tran, model_list, features_df.values.tolist(), False, dict_chain,
                                     data_org_pred, list(features_df.columns), features, thread_num)
        predicted_class = prediction.predict_fun_parallel()
        return [True, predicted_class]
    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tAn error occurred during the prediction step.".format(pid, jobid,
                                                                                                          user))
        return [False, 'An error occurred during the prediction step.']

def parse_data(data_filename, delimiter, user='unknown', jobid=0, pid=0):
    """
    Parses dataset and splits it into Features, sample_name and data lists, expecting both feature and sample headers
    :param data_filename: dataset filename
    :param delimiter: the kind of delimiter with values "," or "\t"
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :return: a list of three lists, [features, data, samples].
    """
    num_of_lines = 0
    features = list()
    data = list()
    samples = list()
    with open(data_filename) as data_fname:
        for line in csv.reader(data_fname, delimiter=delimiter):
            if num_of_lines == 0:
                for j, value in enumerate(line):
                    if j > 0:
                        samples.append(value.strip())
            else:
                data.append([])
                for j, value in enumerate(line):
                    if j == 0:
                        features.append(value.strip())
                    else:
                        if value != '' and value != "#VALUE!":
                            data[num_of_lines - 1].append(float(value))
                        else:
                            data[num_of_lines - 1].append(-1000)
            num_of_lines += 1
    logging.info('PID:{}\tJOB:{}\tUSER:{}\tData were successfully parsed!'.format(pid, jobid, user))
    return [features, data, samples]


def parse_selected_features(features_filename, delimiter, user='unknown', jobid=0, pid=0):
    """
    Parses the selected features filename.

    Args:
        features_filename (string): the selected features filename, one line tab seperated values
        delimiter (string): the kind of delimiter with values "," or "\t"
        user (string): THis job's username
        jobid (integer): this job's id in biomarkers_jobs table
        pid (integer): this job's PID

    Returns:
        features (list): the list of the selected features
    """
    features = list()
    try:
        with open(features_filename) as features_fname:
            for line in csv.reader(features_fname, delimiter=delimiter):
                for i in range(len(line)):
                    features.append(line[i].strip())  # returns a list of one string eg. ['1 2 3']
        features = list(map(int, features[0].split()))  # returns a list of ints eg. [1,2,3]
        logging.info('PID:{}\tJOB:{}\tUSER:{}\tFeatures were successfully parsed!'.format(pid, jobid, user))
        return features
    except Exception:
        logging.exception('PID:{}\tJOB:{}\tUSER:{}\tEmpty selected features file provided!'.format(pid, jobid, user))


def find_delimiter(dataset_filename):
    """
    Figures out which delimiter is being used in given dataset.

    Args:
        dataset_filename (string): the dataset filename

    Returns:
        (string): "," if CSV content, "\t" if TSV content.
    """
    with open(dataset_filename, 'r') as handle:
        head = next(handle)
    if "\t" in head:
        return "\t"
    elif "," in head:
        return ","
    elif "," and "\t" in head:  # The case where the comma is the decimal separator (greek system)
        return "\t"
    else:
        return ";"


def create_feature_list(dataset):
    """
    Creates a feature list with dummy names for a given dataset.

    Args:
        dataset (list): a list of lists

    Returns:
        (list): a one dimensional list with strings "Feature_0", "Feature_1", etc.
    """
    n = len(dataset)
    return ["Feature_" + str(i) for i in range(n)]


def create_samples_list(dataset):
    """
    Creates a samples list with dummy names for a given dataset.

    Args:
        dataset (list): a list of lists

    Returns:
        (list): a one dimensional list with strings "Sample_0", "Sample_1", etc.
    """
    n = len(dataset[0])
    return ["Sample_" + str(i) for i in range(n)]


def perform_missing_value_imputation(dataset_initial, averages, missing_imputation_method, user='unknown', jobid=0,
                                     pid=0):
    """
    Perform missing values imputation.

    Args:
        dataset_initial: the initial dataset (a list of lists)
        missing_imputation_method (integer): 1 for average imputation, 2 for KNN-impute
        averages (list): list of averages per feature, used for imputation
        user (string): THis job's username
        jobid (integer): this job's id in biomarkers_jobs table
        pid (integer): this job's PID

    Returns: a list with the final dataset (list of lists) and the output message (string).
    """
    # missing values imputation
    # KNN-impute
    dataset_initial = list(map(list, zip(*dataset_initial)))
    for i in range(len(dataset_initial)):
        for j in range(len(dataset_initial[0])):
            if dataset_initial[i][j] == '' or dataset_initial[i][j] == -1000:
                dataset_initial[i][j] = np.NaN
    dataset = knn_impute_optimistic(np.asarray(dataset_initial), np.isnan(np.asarray(dataset_initial)), k=3)
    dataset = list(map(list, zip(*dataset)))
    logging.info("PID:{}\tJOB:{}\tUSER:{}\tKNN imputation method was used!".format(pid, jobid, user))
    return dataset


def normalize_dataset(dataset_initial, minimums, maximums, normalization_method, user='unknown', jobid=0, pid=0):
    """
    Normalize the Test dataset, according to Training parameters.

    Args:
        dataset_initial: the initial dataset (a list of lists)
        minimums (list): a list with the minimum values of each feature
        maximums (list): a list with the maximum values of each feature
        normalization_method (integer): 1 for arithmetic sample-wise normalization, 2 for logarithmic normalization
        user (string): THis job's username
        jobid (integer): this job's id in biomarkers_jobs table
        pid (integer): this job's PID

    Returns: if method 1 selected a list with the normalized dataset and the output message, else for method 2
            logged data are returned along with the output message.
    """
    if normalization_method == 1:
        # Arithmetic sample-wise normalization
        outdata_data = [[]]*len(dataset_initial)
        for i in range(len(dataset_initial)):
            outdata_data[i] = [0]*len(dataset_initial[0])
            for j in range((len(dataset_initial[0]))):
                if dataset_initial[i][j] != '' and dataset_initial[i][j] != -1000:
                    if maximums[i] - minimums[i] == 0:
                        outdata_data[i][j] = dataset_initial[i][j]
                    else:
                        outdata_data[i][j] = 0 + (float(dataset_initial[i][j]) - minimums[i]) / float(
                            maximums[i] - minimums[i])
        logging.info("PID:{}\tJOB:{}\tUSER:{}\tArithmetic normalization was used!".format(pid, jobid, user))
        return outdata_data


def average_duplicate_measurements(dataset_initial, markers, user='unknown', jobid=0, pid=0):
    """
    Average duplicate measurements.

    Args:
        dataset_initial: the initial dataset, a list of lists
        markers: input biomarkers
        user (string): THis job's username
        jobid (integer): this job's id in biomarkers_jobs table
        pid (integer): this job's PID

    Returns: a list of two lists, data (a list of lists) and markers (a single list).
    """
    dataset = {}
    dict_of_occurences = {}
    num_of_elements = 0
    for i in range(len(dataset_initial)):
        if markers[i] not in dataset:
            dict_of_occurences[markers[i]] = 1
            dataset[markers[i]] = list()
            for j in range(len(dataset_initial[0])):
                if dataset_initial[i][j] != -1000 and dataset_initial[i][j] != '':
                    dataset[markers[i]].append(float(dataset_initial[i][j]))
                else:
                    dataset[markers[i]].append(0)
        else:
            dict_of_occurences[markers[i]] += 1
            for j in range(len(dataset_initial[0])):
                if dataset_initial[i][j] != -1000 and dataset_initial[i][j] != '':
                    dataset[markers[i]][j] = dataset[markers[i]][j] + float(dataset_initial[i][j])
        num_of_elements += 1

    for key in dataset:
        for j in range(len(dataset[key])):
            dataset[key][j] = dataset[key][j] / dict_of_occurences[key]
    data = list()
    markers = list()
    num_of_markers = 0
    for key, vals in dataset.items():
        data.append([])
        markers.append(key)
        for i in range(len(vals)):
            data[num_of_markers].append(vals[i])
        num_of_markers += 1
    logging.info("PID:{}\tJOB:{}\tUSER:{}\tAveraging duplicate measurements completed successfully!".format(pid, jobid,
                                                                                                            user))
    return [data, markers]


def print_data(data, markers, labels, folder_name, filename):
    """
    Writes data and labels to a file.

    Args:
        data: input data (list of lists)
        markers: input biomarkers (list)
        labels: input labels (list)
        folder_name: output folder
        filename: output filename

    Returns: doesn't return anything, only writes labels and data to a file.
    """
    with open(folder_name + filename, 'w') as file:
        message = ''
        for i in range(len(data[0])):
            message = message + '\t' + labels[i]
        message += '\n'
        for i in range(len(data)):
            message += markers[i]
            for j in range(len(data[0])):
                message += '\t' + str(data[i][j])
            message += '\n'
        file.write(message)


def parse_selected_features_string(astring):
    """
    Parses a string and strips it from commas or newline characters.

    Args:
        astring: the input string with comma separated or newline separated values

    Returns:
        A list with the substrings of the original string.
    """
    if "," in astring:
        return astring.split(",")
    elif "\\n" in astring:
        return astring.split("\\n")
    else:
        # raise ValueError("The string doesn't contain comma separated values or newline separated values !")
        return astring


def has_negatives(list_of_lists):
    """
    Checks if a list of lists contains negative numbers.

    Args:
        list_of_lists (list): the input list

    Returns:
        (boolean): True if yes, False if no

    """
    for x in list_of_lists:
        for a in x:
            if a < 0:
                return True
    return False


def has_negatives_single_list(alist):
    """
    Checks if a single list contains negative numners.

    Args:
        alist (list): the input list

    Returns:
        (boolean): True if yes, False if no
    """
    for x in alist:
        if x < 0:
            return True
    return False


def preprocess_data(input_dataset, maximums, minimums, features_filename,
                    missing_imputation_method, normalization_method, output_folder_name, data_been_preprocessed_flag,
                    variables_for_normalization, has_features_header, has_samples_header, user='unknown',
                    jobid=0, pid=0):
    """
    A script that preprocesses the data. Parse the input files and create a list with the dataset and its features
    Rearrange the dataset in order to align the testing features to the training features order, also perform the
    preprocessing steps if preprocessing is selected

    Args:
        input_dataset: the initial dataset to be preprocessed
        variables_for_normalization: the string with the names of the selected genes that with which the
        geometric mean will be calculated, separated with commas or "\n"
        output_folder_name: the output folder name
        maximums: (list) a list with the maximum values of each feature
        minimums: (list) a list with the minimum values of each feature
        averages_filename: (String) the filename with the average values for each sample
        features_filename: (String) the filename with the indexes of the selected features extracted from the
        training step
        normalization_method: (Integer) 1 for arithmetic, 2 for logarithmic
        missing_imputation_method: (Integer) 1 for average imputation, 2 for KNN imputation
        data_been_preprocessed_flag: (Integer) 0 if data haven't been preprocessed and 1 if they have
        has_features_header: 1 if it has features header, 0 if it doesn't have
        has_samples_header: 1 if it has samples header, 0 if it doesn't have

        user (string): THis job's username
        jobid (integer): this job's id in biomarkers_jobs table
        pid (integer): this job's PID

    Returns:
        output_dataset: the preprocessed dataset
    """
    try:
        # Find delimiter
        delim = find_delimiter(input_dataset)

        if has_features_header and has_samples_header:
            markers, testdata, samples = parse_data(input_dataset, delim, user, jobid, pid)


    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tError during parsing Testing set for preprocessing.".format(
            pid, jobid, user))
        return [False, "Error during parsing the Dataset. Please contact us for more information", '']
    message = ''
    try:
        # Average duplicate measurements & outlier detection
        if missing_imputation_method != 0:
            [testdata, markers] = average_duplicate_measurements(testdata, markers, user, jobid, pid)
            message += "Duplicate measurements have been averaged successfully!\n"
        else:
            message += "Duplicate measurements have not been averaged!\n"
    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tPreprocessing raised the exception during averaging.".format(
            pid, jobid, user))
        return [False, "Preprocessing raised the exception during averaging. Please contact us "
                       "for more information", '']

    try:
        features_training = []
        with open(features_filename, 'r') as features_file:
            for line in features_file:
                for feature in line.split(','):
                    features_training.append(str(feature).strip())

        # Filter the Testset so it has the same features and in the same order as the Training set
        new_data = []  # list to hold dataset after picking features present in training
        new_features = []  # initialise list to hold feature names after deletion
        column_count = len(testdata[0])  # counter for iterating on a feature across observations

        for i, feature in enumerate(features_training):  # match each feature in predict with training
            flag_found = 0  # flag to check if training feature was found in predict feature list
            for j, feat_values in enumerate(testdata):  # get the row index of predict data to be picked for matching
                # feature
                if feature == markers[j]:  # if feature names match
                    flag_found = 1
                    new_data.append(feat_values)  # list of list to hold dataset after matching features
                    new_features.append(markers[j])  # 1st value in list is feature
                    break

            # check if training features not in predict then add null value
            if flag_found == 0:
                logging.info("PID:{}\tJOB:{}\tUSER:{}\tTraining feature not found in predict. Adding null value "
                             "for {}".format(pid, jobid, user, features_training[i]))
                new_data.append([float("nan") for _ in range(column_count)])  # list of list to hold dataset after
                # matching features
                new_features.append(feature)  # training feature as not found in predict data

        logging.info("PID:{}\tJOB:{}\tUSER:{}\tFeatures successfully matched with training features".format(pid, jobid,
                                                                                                            user))
        # return [new_data, new_features]
    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tException during feature selection!".format(pid, jobid, user))
        return [False, "Exception during feature selection", '']

    if data_been_preprocessed_flag == 0:
        try:
            # Perform missing values imputation
            if missing_imputation_method == 2:
                newAverages = []
                dataset_imputed = perform_missing_value_imputation(new_data, newAverages, missing_imputation_method,
                                                                   user, jobid, pid)
            else:
                return [False, "Preprocessing raised the exception No imputation method provided. Please contact "
                               "us for more information", '']
        except Exception:
            logging.exception("PID:{}\tJOB:{}\tUSER:{}\tPreprocessing raised the exception during imputation.".format(
                pid, jobid, user))
            return [False, "Preprocessing raised the exception during imputation. Please check if you selected the same"
                           " imputation method as the training step, or contact us for more information",
                    '']

        try:
            # Catch the cases where the user inputs only one element with a comma/newline at the end
            if '' in variables_for_normalization and not isinstance(variables_for_normalization, str) and \
                    not isinstance(variables_for_normalization, unicode):
                variables_for_normalization.pop()

            # Creating the list of indexes of the selected features
            variables_for_normalization_nums = list()
            if variables_for_normalization:
                if isinstance(variables_for_normalization, list):
                    for variable in variables_for_normalization:
                        try:
                            variable_index = new_features.index(variable)
                            variables_for_normalization_nums.append(variable_index)
                        except Exception:
                            logging.exception("PID:{}\tJOB:{}\tUSER:{}\tThe biomarker(s) provided are not in the "
                                              "list of the biomarkers of the input file!".format(pid, jobid, user))
                            return [False, "The biomarker(s) provided gor normalization are not in the list of the "
                                           "biomarkers of the input file! Try again!", '']
                elif isinstance(variables_for_normalization, str):
                    try:
                        variable_index = new_features.index(variables_for_normalization)
                        variables_for_normalization_nums.append(variable_index)
                    except Exception:
                        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tThe biomarker(s) provided are not in the list of "
                                          "the biomarkers of the input file!".format(pid, jobid, user))
                        return [False, "The biomarker(s) provided gor normalization are not in the list of the "
                                       "biomarkers of the input file! Try again!", '']

            # Perform normalization of dataset
            if normalization_method == 1:
                if maximums and minimums:
                    message += " "
                    normalized_mv_imputed_dataset = normalize_dataset(dataset_imputed, minimums, maximums,
                                                                      normalization_method, user, jobid, pid)
                    dataset_imputed = normalized_mv_imputed_dataset
                else:
                    message += " No normalization took place."
                    inputs = dataset_imputed


        except Exception:
            logging.exception("PID:{}\tJOB:{}\tUSER:{}\tPreprocessing raised the exception during "
                              "normalization.".format(pid, jobid, user))
            return [False, "Preprocessing raised an exception during normalization. Please contact us "
                           "for more information", '']

        # Print output message to info.txt file
        with open(output_folder_name + "info.txt", "w") as handle:
            handle.write(message)

        try:
            # write preprocessed data to file
            print_data(dataset_imputed, new_features, samples, output_folder_name, "preprocessed_dataset_{}.tsv".format(
                jobid))

        except Exception:
            logging.exception("PID:{}\tJOB:{}\tUSER:{}\tPreprocessing raised the exception during printing "
                              "data.".format(pid, jobid, user))
            return [False, "Preprocessing raised the exception during printing data. Please contact us "
                           "for more information", '']
        new_data = dataset_imputed
    return [True, new_data, new_features]


def preprocess_specific(input_dataset, delimiter, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    src_path = os.path.dirname(os.path.realpath(__file__)) + os.sep + "src" + os.sep
    umami_significant = open(src_path + 'umami_statistical_features.txt', 'r')
    umami_features = umami_significant.read()
    umami_features = umami_features.split('\n')
    umami_significant.close()
    df = pd.read_csv(input_dataset, delimiter=delimiter)
    df = df[df.columns.drop(list(df.filter(regex='Unnamed')))]
    df.columns = df.columns.str.replace("-", "_")
    df.columns = df.columns.str.replace(".", "_")
    df = df.rename(
        columns={'nARing': 'nARing_1', 'n5ARing': 'n5ARing_1', 'nAHRing': 'nAHRing_1', 'n5AHRing': 'n5AHRing_1'})

    df = df[umami_features[:-1]]

    df = df.astype(object).T
    xx = df.dropna(axis=0, how='all')
    df = xx.dropna(axis=1, how='all')
    df.values[df.values > 1.9399999999999998e+33] = 1.9399999999999998e+33
    df.to_csv(output_folder + 'preprocessed_data.txt', sep='\t')

    path = output_folder + 'preprocessed_data.txt'
    return path



if __name__ == "__main__":

    testset_filename1 = sys.argv[1]

    src_path = os.path.dirname(os.path.realpath(__file__)) + os.sep + "src" + os.sep

    maximums_filename1 = src_path  + 'maximums.txt'
    minimums_filename1 = src_path  + 'minimums.txt'
    features_filename1 = src_path + 'features_list.txt'
    missing_imputation_method1 = 2
    normalization_method1 = 1
    model_filename1 = src_path  + 'models_3_5.zip'
    selection_flag1 = 2
    data_been_preprocessed_flag1 = 0
    has_features_header1 = 1
    has_samples_header1 = 1

    training_labels_filename1 = src_path  + 'training_labels.txt'
    length_of_features_from_training_filename1 = src_path  + 'length_of_features_from_training.txt'
    tstamp = time.strftime('%Y_%m_%d_%H_%M')
    output_folder1 = os.getcwd() + os.sep + 'Output_folder_' + str(tstamp) + os.sep
    selected_comorbidities_string1 = ""
    initLogging()

    delim = find_delimiter(testset_filename1)
    dataset = preprocess_specific(testset_filename1, delim, output_folder1)
    ret = run_all(dataset, maximums_filename1, minimums_filename1,
                  features_filename1, missing_imputation_method1, normalization_method1,
                  model_filename1, selection_flag1, data_been_preprocessed_flag1, selected_comorbidities_string1,has_features_header1, has_samples_header1, training_labels_filename1,
                  length_of_features_from_training_filename1, output_folder1)

    logging.info("{}".format(ret[1]))
