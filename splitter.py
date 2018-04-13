import argparse
import sys
import numpy as np
import random


def percentage_split(data_set, percentages):
    """
    This function splits the provided dataset into the provided percentages
    :param data_set: The dataset to be split
    :param percentages: The percentages, in the following order: training, validation and test
    :return: A nested list containing the splitted dataset
    """
    random.shuffle(data_set)
    cdf = np.cumsum(percentages)
    stops = list(map(int, cdf * len(data_set)))
    return [data_set[a:b] for a, b in zip([0] + stops, stops)]

def get_data_sets(model_data, train_perc, vali_perc, test_perc, output_folder):
    """
    This function aids to split a provided dataset in the train, validation and
    test sets based on the provided percentages.
    :param model_data: The data that needs to be split
    :param train_perc: Size of the training set(%)
    :param vali_perc: Size of the validation set(%)
    :param test_perc: Size of the test set(%)
    :param output_folder: The folder in which the sets needs to be saved
    """
    percs = [train_perc, vali_perc, test_perc]
    total_perc = round(sum(percs), 1)  # round to avoid 99,999...etc
    if total_perc != 1:
        print('[ERROR] Percentages do not add up to 1')
        sys.exit(1)
    else:
        try:
            data_set = np.load(model_data)
            train, vali, test = percentage_split(data_set, percs)
            np.save(output_folder + 'training.npy', train)
            np.save(output_folder + 'validation.npy', vali)
            np.save(output_folder + 'test.npy', test)
            print("[INFO] total data size: {}".format(len(data_set)))
            print('[INFO] train size: {}'.format(len(train)))
            print('[INFO] validate size: {}'.format(len(vali)))
            return train, vali, test
        except Exception as e:
            print("[ERROR] Numpy cannot load model data")
            print(str(e))
            sys.exit(1)

# Parsing the arguments provided by the user
parser = argparse.ArgumentParser()
parser.add_argument('-d','--data', type=str, help='The dataset that needs to be split', required = True)
parser.add_argument('-r','--training', type=float, help='% of the data that needs to be used for training', required = True)
parser.add_argument('-v','--validation', type=float, help='% of the data that needs to be used for validation', required = True)
parser.add_argument('-t','--test', type=float, help='% of the data that needs to be used for testing', required = True)
parser.add_argument('-o','--output_folder', type=str, help='output_folder', required = True)
args = parser.parse_args()

# Using the percentages to obtain the training, validation and test set
get_data_sets(args.data, args.training, args.validation, args.test, args.output_folder)

