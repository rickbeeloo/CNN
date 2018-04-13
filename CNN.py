import tensorflow as tf
import timeit
import traceback
import csv
import sys
import numpy as np

class CNN:
    """
    This class forms the base of the 3D CCN network
    """

    def __init__(self, img_size_px, slice_count, n_classes, batch_size, keep_rate,
                 hm_epochs, gpu = True, model_name = 'model'):

        self.gpu = gpu
        self.hm_epochs = hm_epochs
        self.img_size_px = img_size_px
        self.slice_count = slice_count
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.keep_rate = keep_rate
        self.model_name = model_name

    def __conv3d(self, x, W):
        """
        This function computes a 3-D convolution.
        :param x: A float placeholder (tensor)
        :param W: The weights that should be used
        :return: A 3D CNN based comprising the provided weights
        """
        return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

    def __maxpool3d(self, x):
        """
        This function performs 3D max pooling
        :param x: A float placeholder (tensor)
        :return: The 3D pool
        """
        return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

    def __get_weights(self):
        """
        This function will return the weight used in the model
        :return: The model
        """
        return {'W_conv1': tf.Variable(tf.random_normal([3, 3, 3, 1, 32])),
                   #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
                   'W_conv2': tf.Variable(tf.random_normal([3, 3, 3, 32, 64])),
                   #                                  64 features
                   'W_fc': tf.Variable(tf.random_normal([54080, 1024])),
                   'out': tf.Variable(tf.random_normal([1024, self.n_classes]))}

    def __get_biases(self):
        """
        This function returns the biases used in the model
        :return: The biases
        """
        return {'b_conv1': tf.Variable(tf.random_normal([32])),
                  'b_conv2': tf.Variable(tf.random_normal([64])),
                  'b_fc': tf.Variable(tf.random_normal([1024])),
                  'out': tf.Variable(tf.random_normal([self.n_classes]))}


    def __convolutional_neural_network(self, x):
        """
        This function uses the conv3d, maxpool3d, get_weights, and get_biases functions to
        actually build the model, which can be trained hereafter (using the train_neural_network
        function)
        :param x: A float placeholder
        :return:A 3D CNN model which can be trained.
        """
        #                # 5 x 5 x 5 patches, 1 channel, 32 features to compute.
        weights = self.__get_weights()
        biases =  self.__get_biases()

        # reshaping the image to the provided pixels (see init)
        x = tf.reshape(x, shape=[-1, self.img_size_px, self.img_size_px, self.slice_count, 1]) #image x, y, z
        conv1 = self.__maxpool3d(tf.nn.relu(self.__conv3d(x, weights['W_conv1']) + biases['b_conv1']))
        conv2 = self.__maxpool3d(tf.nn.relu(self.__conv3d(conv1, weights['W_conv2']) + biases['b_conv2']))

        fc = tf.reshape(conv2, [-1, 54080])
        fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
        fc = tf.nn.dropout(fc, self.keep_rate)

        output = tf.matmul(fc, weights['out']) + biases['out']
        return output

    def __build_output(self, run_time, accuracy, fitment_perc):
        """
        This function will return all parameters and the newly
        obatined parameters from a training run.
        :param run_time: The run time in seconds
        :param accuracy:  The accuracy of the training
        :param fitment_perc: The fitment percentage (decimal)
        :return: The model parameters and statistics for training
        """
        return {
            'run_time' : run_time,
            'accuracy' : accuracy,
            'fitment_percent' : fitment_perc,
            'number_epochs' : self.hm_epochs,
            'img_size_px' : self.img_size_px,
            'slice_count' : self.slice_count,
            'batch_size' : self.batch_size,
            'keep_rate': self.keep_rate
        }


    def __get_processor(self):
        """
        This function can be used to switch between CPU and GPU
        :return: configuration indicating which processor should be used.
        """
        if self.gpu:
            return tf.ConfigProto() #just return default config
        else:
            return tf.ConfigProto(
                device_count={'GPU': 0}
            )

    def __reverse_label(self, labels):
        """
        Reverse the input format for tensorflow to human readable format
        :param labels:
        :return:
        """
        return [1 if label[1] == 1 else 0 for label in labels]


    def __write_stats(self, ids, labels, predicted, output_path):
        """
        This function writes the testing stats to the given output path
        :param ids: The patient ids
        :param labels: The orginal labels of the patients (1 = sick, 0 = norma)
        :param predicted:  The predicted class for the patient
        :param output_path: The output path to which the results should be written
        """
        with open(output_path,'w') as out_file:
            header = ['id','label','predicted']
            writer = csv.writer(out_file, delimiter='\t')
            writer.writerows([header])
            writer.writerows(zip(*[ids, self.__reverse_label(labels), predicted]))

    def test_neural_network(self, test_data, model_name, model_folder, output_path = None):
        """
        This function can be used to test the neural network on a training set
        :param test_data: The test data (obtained from splitter.py)
        :param model_name: The name of the model that should be used
        :param model_folder: The folder in which the model is saved
        :param output_path: The path to which the output should be written (see write stats)
        """
        x = tf.placeholder('float')
        y = tf.placeholder('float')
        prediction = self.__convolutional_neural_network(x)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()

            for epoch in range(self.hm_epochs):
                try:
                    # this will load the trained model
                    saver.restore(sess,  model_folder + model_name + '.ckpt')
                except Exception as e:
                    print(str(e))

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            test_data = np.load(test_data)
            ids, features, labels =[],[],[]

            for item in test_data:
                data = item[0]
                label = item[1]
                id = item[2]
                ids.append(id)
                features.append(data)
                labels.append(label)

            test_x = np.array(features)
            test_y = np.array(labels)

            # this will predicted the classes and will write the given
            # and predicted classes to the output file (if specified)
            if output_path:
                predicted = sess.run(tf.argmax(test_y, 1), feed_dict={x: test_x})
                self.__write_stats(ids, labels, predicted, output_path)
                print('[INFO] test stats written to: {}'.format(output_path))

            print('[INFO] Accuracy:', accuracy.eval({x: test_x, y: test_y}))

    def __load_data(self, train_data, validation_data):
        """
        This function can be used to load the training and validation data
        :param train_data: The training data that should be loaded
        :param validation_data: The validation data that should be loaded
        :return:
        """
        try:
            train_data = np.load(train_data)
            validation_data = np.load(validation_data)
            return train_data, validation_data
        except FileNotFoundError as e:
            print('[ERROR] The file {} was not found!'.format(e.filename))
            sys.exit(1)

    def train_neural_network(self, train_data, validation_data, output_folder = '', return_output = False):
        """
        This function can be used to train the neural network
        :param train_data: The training data (obtained from splitter.py)
        :param validation_data: The validation data (obtained from splitter.py)
        :param output_folder: The folder to which the model should be written
        :param return_output: Indicating whether the stats should be returned or not
        :return: The statistics for the training (see build_output function)
        """
        start = timeit.default_timer()
        x = tf.placeholder('float')
        y = tf.placeholder('float')
        train_data, validation_data = self.__load_data(train_data, validation_data)
        prediction = self.__convolutional_neural_network(x)
        print("finished prediction")
        cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
        try:
            saver = tf.train.Saver()
            with tf.Session(config = self.__get_processor()) as sess:
                sess.run(tf.global_variables_initializer())

                successful_runs = 0
                total_runs = 0
                accuracy = 0

                for epoch in range(self.hm_epochs):
                    epoch_loss = 0
                    for data in train_data:
                        total_runs += 1
                        try:
                            X = data[0]
                            Y = data[1]
                            _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                            epoch_loss += c
                            successful_runs += 1
                        except Exception as e:
                            pass


                    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                    current_accuracy = accuracy.eval({x: [i[0] for i in validation_data], y: [i[1] for i in validation_data]})
                    print('[INFO] Epoch', epoch + 1, 'completed out of', self.hm_epochs, 'loss:', epoch_loss)
                    print('[INFO] Accuracy:', current_accuracy )

                finish_acc = accuracy.eval({x: [i[0] for i in validation_data], y: [i[1] for i in validation_data]})
                print('[INFO] Finished Accuracy:',finish_acc )
                print('[INFO] fitment percent:', successful_runs / total_runs)

                run_time = timeit.default_timer() - start
                print('[INFO] runtime: {}'.format(run_time))
                if return_output:
                    return self.__build_output(run_time, finish_acc, successful_runs / total_runs)
                save_path = saver.save(sess, output_folder + self.model_name + '.ckpt')
                print("[INFO] Model saved in path: %s" % save_path)

            tf.reset_default_graph()
        except ValueError:
            print("[ERROR] Could not save the model!")
        except Exception as e:
            print('[ERROR] an unexpected errro was encountered')





