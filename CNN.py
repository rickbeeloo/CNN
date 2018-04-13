import tensorflow as tf
import timeit
import argparse
import traceback
import numpy as np

class CNN:

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
        return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

    def __maxpool3d(self, x):
        return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

    def __get_weights(self):
        return {'W_conv1': tf.Variable(tf.random_normal([3, 3, 3, 1, 32])),
                   #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
                   'W_conv2': tf.Variable(tf.random_normal([3, 3, 3, 32, 64])),
                   #                                  64 features
                   'W_fc': tf.Variable(tf.random_normal([54080, 1024])),
                   'out': tf.Variable(tf.random_normal([1024, self.n_classes]))}

    def __get_biases(self):
        return {'b_conv1': tf.Variable(tf.random_normal([32])),
                  'b_conv2': tf.Variable(tf.random_normal([64])),
                  'b_fc': tf.Variable(tf.random_normal([1024])),
                  'out': tf.Variable(tf.random_normal([self.n_classes]))}


    def __convolutional_neural_network(self, x):
        #                # 5 x 5 x 5 patches, 1 channel, 32 features to compute.
        weights = self.__get_weights()
        biases =  self.__get_biases()

        x = tf.reshape(x, shape=[-1, self.img_size_px, self.img_size_px, self.slice_count, 1]) #imag x, y, z
        conv1 = self.__maxpool3d(tf.nn.relu(self.__conv3d(x, weights['W_conv1']) + biases['b_conv1']))
        conv2 = self.__maxpool3d(tf.nn.relu(self.__conv3d(conv1, weights['W_conv2']) + biases['b_conv2']))

        fc = tf.reshape(conv2, [-1, 54080])
        fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
        fc = tf.nn.dropout(fc, self.keep_rate)

        output = tf.matmul(fc, weights['out']) + biases['out']
        return output

    def __build_output(self, run_time, accuracy, fitment_perc):
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
        if self.gpu:
            return tf.ConfigProto() #just return default config
        else:
            return tf.ConfigProto(
                device_count={'GPU': 0}
            )

    def test_neural_network(self, test_data, model_name, model_folder):
        x = tf.placeholder('float')
        y = tf.placeholder('float')
        prediction = self.__convolutional_neural_network(x)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()

            for epoch in range(self.hm_epochs):
                try:
                    saver.restore(sess,  model_folder + model_name + '.ckpt')
                except Exception as e:
                    print(str(e))

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            test_data = np.load(test_data)
            features, labels = [],[]
            for item in test_data:
                data = item[0]
                id = item[1]
                features.append(data)
                labels.append(id)
            test_x = np.array(features)
            test_y = np.array(labels)
            print(test_x)
            print(test_y)
            print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))


    def train_neural_network(self, train_data, validation_data,output_folder = '', return_output = False):
        start = timeit.default_timer()
        x = tf.placeholder('float')
        y = tf.placeholder('float')
        train_data = np.load(train_data)
        validation_data = np.load(validation_data)
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
                #accuracy_vector = []

                for epoch in range(self.hm_epochs):
                    epoch_loss = 0
                    for data in train_data:
                        total_runs += 1
                        print('total: {}'.format(total_runs))
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
                    #accuracy_vector.append(current_accuracy)
                    print('[INFO] Epoch', epoch + 1, 'completed out of', self.hm_epochs, 'loss:', epoch_loss)
                    print('[INFO] Accuracy:', current_accuracy )

                finish_acc = accuracy.eval({x: [i[0] for i in validation_data], y: [i[1] for i in validation_data]})
                print('[INFO] Finished Accuracy:',finish_acc )
                print('[INFO] fitment percent:', successful_runs / total_runs)


                run_time = timeit.default_timer() - start
                print('[INFO] runtime: {}'.format(run_time))
                print(output_folder)
                save_path = saver.save(sess, output_folder + self.model_name + '.ckpt')
                print("Model saved in path: %s" % save_path)
                if return_output:
                    return self.__build_output(run_time, finish_acc, successful_runs / total_runs)

            tf.reset_default_graph()
        except Exception as e:
            print(e)
            traceback.print_exc()
            print('ERROR')




