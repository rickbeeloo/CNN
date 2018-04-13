from CNN_adjusted import  CNN_adjusted as CNN
from Plot import Plotter
import csv
import copy


# base model params so we do not have to specify them each time
base_model_params = {
        'img_size_px' : 50,
        'slice_count' : 20,
        'n_classes' : 2,
        'batch_size' : 10,
        'keep_rate' : 0.8,
        'hm_epochs' : 10,
        'gpu': True,
        'model_name' : 'model'
    }


def edit_params(variables, values):
    """
    This function can be used to edit the base parameters
    :param variables: The variables that should be modified
    :param values: The corresponding values of the variables
    :return: The updated parameter list
    """
    params = copy.deepcopy(base_model_params)
    for variable, value in zip(variables, values):
        params[variable] = value
    return params


def write_to_file(path, header, data):
    """
    This function write the provided data (column wise) to
    the specified file path
    :param path: The output path of the file
    :param header: The file header
    :param data: The data (nested list in which each item is a column)
    """
    new_line = '\n'
    with open(path,'w') as out_file:
        writer = csv.writer(out_file, delimiter='\t')
        writer.writerows([header])
        writer.writerows(zip(*data))

    print("[INFO] finished writing to: {}".format(path))



def cpu_vs_gpu(epochs, train_data, validation_data, table_output_path = ''):
    """
    This function can be used ot produce to compare the speed (in seconds) between
    the usage of the gpu and cpu.
    :param epochs: A list of epochs which should be used
    :param train_data: The training data (obtained from splitter.py)
    :param validation_data: The validation data (obtained from splitter.py)
    :param table_output_path: The path to which the output should be written
    """
    gpu_results = []
    cpu_results = []
    p = Plotter()
    for epoch in epochs:
        gpu_params = edit_params(['hm_epochs' ,'gpu'], [epoch, True])
        cpu_params = edit_params(['hm_epochs' ,'gpu'], [epoch, False])
        gpu_results.append(CNN().run_model(gpu_params, train_data, validation_data))
        cpu_results.append(CNN().run_model(cpu_params, train_data, validation_data))

    gpu_x, gpu_y = p.get_x_y(gpu_results, 'number_epochs', 'run_time')
    cpu_x, cpu_y = p.get_x_y(cpu_results, 'number_epochs', 'run_time')
    if table_output_path:
        header = ['epoch','gpu_time','cpu_time']
        write_to_file(table_output_path + 'gpu_vs_cpu.txt',header, [gpu_x, gpu_y, cpu_y])
    p.double_line([gpu_x,cpu_x], [gpu_y, cpu_y], ['GPU','CPU'])

if __name__ == '__main__':
    cpu_vs_gpu([1,5,10,15,20,25,30,35,40,45,50], 'dat a/training.npy', 'data/validation.npy', 'data/')