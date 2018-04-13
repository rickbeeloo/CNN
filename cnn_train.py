import argparse
from CNN import CNN

# defining the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-s','--image_size', type=int, help='The pixels to which the image should be adjusted to', default=50)
parser.add_argument('-c','--slice_count', type=int, help='Number of slices to normalize to', default = 20)
parser.add_argument('-a','--n_classes', type=int, help='The number of classes in the dataset', default = 2)
parser.add_argument('-b','--batch_size', type=int, help='The batch size', default = 10)
parser.add_argument('-k','--keep_rate', type=float, help='The batch size', default = 0.8)
parser.add_argument('-e','--epochs', type=int, help='The number of epochs', default = 10)
parser.add_argument('-g','--gpu_usage', type=bool, help='Path to the data needed for validation', default = True)
parser.add_argument('-n','--model_name', type=str, help='The model name', default = 'model')
parser.add_argument('-r','--training_data', type=str, help='Path to the data needed for training', required = True)
parser.add_argument('-v','--validation_data', type=str, help='Path to the data needed for validation', required = True)
parser.add_argument('-o','--output_path', type=str, help='Path to the data needed for validation', required = True)

# parsing the arguments
args = parser.parse_args()

# buidling and training the model
model = CNN(args.image_size, args.slice_count, args.n_classes, args.batch_size, args.keep_rate, args.epochs, args.gpu_usage, args.model_name)
model.train_neural_network(args.training_data, args.validation_data, args.output_path)

