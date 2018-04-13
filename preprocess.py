import numpy as np
import pandas as pd
import argparse
from random import randint
import dicom
from timeit import default_timer as timer
import os
import matplotlib.pyplot as plt
import cv2
import math
import sys

class PreProcessor:
    """
    This class functions as a lung scan image processor
    """

    def __init__(self, img_size_px, slice_count, model_name):
        """
        :param img_size_px: Number of pixels to which the image should be reduced
        :param slice_count: The number of slices that should be used
        :param model_name: The output name for the processed data
        """
        self.image_size_px = img_size_px
        self.hm_slices = slice_count
        self.model_name = model_name
        self.labels = None
        self.data_dir = None

    def chunks(self, l, n):
        """
        This function splits a list into n parts
        :param l: The list that needs to be split
        :param n: The number of wanted chunks
        :return: a list containing the wanted chunks
        """
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def mean(self, a):
        """
        Calculates the mean
        :param a: The list for which the mean should be calculated
        :return: The mean(average) of the list
        """
        return sum(a) / len(a)

    def load_data(self, image_dir, metadata_path):
        """
        This function loads the data which needs to be processed
        :param image_dir: The path to the directory containing the lung images
        :param metadata_path: The path to the image metadata file
        """
        self.data_dir = image_dir
        self.labels = pd.read_csv(metadata_path, index_col=0)


    def process_data(self):
        """
        This function processes the image data
        """
        start = timer()
        much_data = []
        patients = os.listdir(self.data_dir)
        for num, patient in enumerate(patients):
            if num % 100 == 0:
                print("[INFO] #{}".format(num))
                print('[INFO] {}%'.format(num / len(patients) * 100))
                print("[INFO] time elapsed: {} minutes".format(timer() - start / 60))
            try:
                img_data, label = self.process_patient(patient)
                much_data.append([img_data, label])
            except KeyError as e:
                print('[INFO] Patient {} does not have metadata coupled'.format(num))
                print('[INFO] This will not cause any problems, but the patient will not be included')
        file_name = '{}.npy'.format(self.model_name)
        np.save(file_name, much_data)


    def process_example(self, type = 'sick', patient_id = None):
        """
        This function can be used to randomly obtain a lung scan from a
        sick or healthy person to get a feel of what the data looks like.
        :param type: The example image you want, either sick or normal
        :param patient_id:
        """
        if patient_id:
            self.process_patient(patient_id, visualize=True)
        else:
            example_found = False
            allowed = ['healthy','sick'] #0 = normal, #1 = sick
            if type not in allowed:
                print("[ERROR] Type has to be either 'sick'  or 'healthy'")
            else:
                wanted_label = allowed.index(type)
                patients = os.listdir(self.data_dir)
                while example_found == False:
                    try:
                        rand_numb = randint(0, len(patients))
                        patient = patients[rand_numb]
                        label = self.labels.get_value(patient, 'cancer')
                        if label == wanted_label:
                            print(patient)
                            return self.process_patient(patient,visualize = True )
                    except KeyError:
                        continue


    def process_patient(self, patient, visualize = False):
        """
        This function processes the data for an individual patient
        :param patient: The patient id
        :param visualize: A boolean to indicate whether the patients scan should be visualised
        :return: Numpy array containing the processed data
        """
        label = self.labels.get_value(patient, 'cancer')
        path = self.data_dir + patient
        try:
            slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
            slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

            new_slices = []
            slices = [cv2.resize(np.array(each_slice.pixel_array), (self.image_size_px, self.image_size_px)) for each_slice in slices]

            chunk_sizes = math.ceil(len(slices) / self.hm_slices)
            for slice_chunk in self.chunks(slices, chunk_sizes):
                slice_chunk = list(map(self.mean, zip(*slice_chunk)))
                new_slices.append(slice_chunk)

            if len(new_slices) == self.hm_slices - 1:
                new_slices.append(new_slices[-1])

            if len(new_slices) == self.hm_slices - 2:
                new_slices.append(new_slices[-1])
                new_slices.append(new_slices[-1])

            if len(new_slices) == self.hm_slices + 2:
                new_val = list(map(self.mean, zip(*[new_slices[self.hm_slices - 1], new_slices[self.hm_slices], ])))
                del new_slices[self.hm_slices]
                new_slices[self.hm_slices - 1] = new_val

            if len(new_slices) == self.hm_slices + 1:
                new_val = list(map(self.mean, zip(*[new_slices[self.hm_slices - 1], new_slices[self.hm_slices], ])))
                del new_slices[self.hm_slices]
                new_slices[self.hm_slices - 1] = new_val

            if visualize:
                fig = plt.figure()
                for num, each_slice in enumerate(new_slices):
                    y = fig.add_subplot(4, 5, num + 1)
                    y.imshow(each_slice,cmap='gray')
                plt.show()


            if label == 1:
                label = np.array([0, 1])
            elif label == 0:
                label = np.array([1, 0])
            return np.array(new_slices), label
        except FileNotFoundError:
            print('[ERROR] The slices could not be retrieved check the provided image path')
            sys.exit(1)




# Parsing the arguments provided by the user
parser = argparse.ArgumentParser()
parser.add_argument('-s','--image_size', type=int, help='The pixels to which the image should be adjusted to', default = 50)
parser.add_argument('-c','--slice_count', type=int, help='Number of slices to normalize to', default = 20)
parser.add_argument('-o','--output_path', type=str, help='The location to which the output should be written', required = True)
parser.add_argument('-i','--image_path', type=str, help='Path to the folder containing the images', required = True)
parser.add_argument('-m','--metadata_path', type=str, help='Path to the image metadata file', required = True)
args = parser.parse_args()

# Processing the data provided by the user
processor = PreProcessor(args.image_size, args.slice_count, args.output_path)
processor.load_data(args.image_path, args.metadata_path)
processor.process_data()