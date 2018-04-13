import multiprocessing as mp
from multiprocessing import Process
import sys
from CNN import  CNN

class CNN_adjusted:
    """
    This class can be substituted by the CNN class in the code whenever the used GPU card has enough
    memory to save multiple instances of the data. If not  (in our case for example a GPU card with 2gb)
    a process needs to be simulated to be able to terminate it after every iteration as tensorflow will
    not wipe the GPU memory automatically until the full code is done running.
    """


    def model(self, model_params, return_dict, train_data, validation_data):
        """
        This functions builds the CNN based on the provided parameters
        :param model_params: The parameters that have to be used for the model
        :param return_dict: A dictionary in which the results should be saved
        :return: results in a dictionary
        """
        model = CNN(**model_params)
        output = model.train_neural_network(train_data, validation_data,output_folder= 'data/temp/', return_output = True)
        return_dict['data'] = output

    def run_model(self, model_params, train_data, validation_data):
        """
        This function start a process and terminates its afterwards to clean
        the GPU memory for the next iteration.
        :param model_params: The parameters that have to be used for the model
        :return: The data returned by the CNN model
        """
        manager = mp.Manager()
        return_dict = manager.dict()
        p = Process(target=self.model, args=(model_params, return_dict, train_data, validation_data))
        jobs = [p]
        p.start()
        for job in jobs:
            job.join()
        p.terminate()
        if return_dict:
            return return_dict['data']
        else:
            print('[ERROR] Could not finish the process')
            sys.exit(1)


