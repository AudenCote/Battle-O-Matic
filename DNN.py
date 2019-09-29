from matrix import *
import random
import pickle
import numpy

class NeuralNetwork:

    def __init__(self, input_nodes, output_nodes, rate):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

        self.hidden_layers = 0
        self.hidden_nodes_array = []

        self.weights = []
        self.biases = []

        self.hidden_biases = []

        self.learning_rate = rate

    def hidden_layer(self, hidden_nodes):
        self.hidden_layers += 1
        self.hidden_nodes_array.append(hidden_nodes)

    def initialize_weights(self):
        self.weights.append(random_uniform([self.hidden_nodes_array[0], self.input_nodes]))
        self.biases.append(random_uniform([self.hidden_nodes_array[0], 1]))

        for i in range(self.hidden_layers - 1):
            self.weights.append(random_uniform([self.hidden_nodes_array[i+1], self.hidden_nodes_array[i]]))
            self.biases.append(random_uniform([self.hidden_nodes_array[i + 1], 1]))

        self.weights.append(random_uniform([self.output_nodes, self.hidden_nodes_array[-1]]))
        self.biases.append(random_uniform([self.output_nodes, 1]))


    def mean_square_error(self, targets, output):
        sum_square_error = 0
        for i, row in enumerate(targets):
            for j, val in enumerate(row):
                sum_square_error += (targets[i][j] - output[i][j])**2

        loss = sum_square_error / len(targets)

        return loss

    def predict(self, input_array, data):
        inputs = unflatten(input_array)
        weights, biases = data

        weights_ih = weights[0]
        weights_ho = weights[-1]
        bias_h = biases[0]
        bias_o = biases[-1]

        hidden1 = matrix_multiplication(weights_ih, inputs)
        hidden1 = elementwise_addition(hidden1, bias_h)
        new_hidden = sigmoid(hidden1)

        hiddens = [new_hidden]
        for i in range(self.hidden_layers - 1):
            new_hidden = matrix_multiplication(weights[i + 1], hiddens[-1])
            new_hidden = elementwise_addition(new_hidden, biases[i + 1])
            new_hidden = sigmoid(new_hidden)

            hiddens.append(new_hidden)

        outputs = matrix_multiplication(weights_ho, hiddens[-1])
        outputs = elementwise_addition(outputs, bias_o)
        outputs = sigmoid(outputs)

        return outputs

    def feed_and_propogate(self, input_array, targets_array, epochs, batch_size, graph):
        input_all = []
        for v, val in enumerate(input_array):
            input_all.append([input_array[v], targets_array[v]])

        batches  = []
        for q, val in enumerate(input_all):
            if q % batch_size == 0:
                batch = input_all[q:q+batch_size]
                batches.append(batch)

        losses = []
        epoch_list = []
        for e in range(epochs):
            epoch_list.append(e + 1)

            for b, batch in enumerate(batches):
                random.shuffle(batch)

                bias_deltas = []
                weights_deltas = []
                for it_pair in batch:

                    sample_bias_deltas = []
                    sample_weights_deltas = []

                    inputs = it_pair[0]
                    targets = it_pair[1]

                    inputs = unflatten(inputs)

                    hidden1 = matrix_multiplication(self.weights[0], inputs)
                    hidden1 = elementwise_addition(hidden1, self.biases[0])
                    new_hidden = sigmoid(hidden1)

                    hiddens = [new_hidden]
                    for i in range(self.hidden_layers - 1):
                        new_hidden = matrix_multiplication(self.weights[i + 1], hiddens[-1])
                        new_hidden = elementwise_addition(new_hidden, self.biases[i+1])
                        new_hidden = sigmoid(new_hidden)

                        hiddens.append(new_hidden)

                    outputs = matrix_multiplication(self.weights[-1], hiddens[-1])
                    outputs = elementwise_addition(outputs, self.biases[-1])
                    outputs = sigmoid(outputs)

                    results_index = int(numpy.argmax(outputs))

                    targets = unflatten(targets)

                    #BACK PROPOGATION

                    loss = NeuralNetwork.mean_square_error(self, targets, outputs)
                    last_errors = elementwise_subtraction(targets, outputs)
                    gradients = derivative_sigmoid(outputs)
                    gradients = elementwise_multiplication(gradients, last_errors)
                    gradients = scalar_multiplication(gradients, self.learning_rate)

                    hidden3_t = transpose(hiddens[-1])
                    weight_ho_deltas = matrix_multiplication(gradients, hidden3_t)

                    sample_weights_deltas.append(weight_ho_deltas)
                    sample_bias_deltas.append(gradients)

                    for i in range(self.hidden_layers - 1):
                        current = self.weights[-(i+1)]
                        new_hidden = hiddens[-(i+2)]

                        current_transposed = transpose(current)

                        last_errors = matrix_multiplication(current_transposed, last_errors)
                        gradient = derivative_sigmoid(hiddens[-(i+1)])
                        gradient = elementwise_multiplication(gradient, last_errors)
                        gradient = scalar_multiplication(gradient, self.learning_rate)

                        new_hidden_transposed = transpose(new_hidden)
                        deltas = matrix_multiplication(gradient, new_hidden_transposed)

                        sample_weights_deltas.append(deltas)
                        sample_bias_deltas.append(gradient)
                        
                    weights1_t = transpose(self.weights[1])
                    hidden1_errors = matrix_multiplication(weights1_t, last_errors)
                    
                    hidden1_gradient = derivative_sigmoid(hidden1)
                    hidden1_gradient = elementwise_multiplication(hidden1_gradient, hidden1_errors)
                    hidden1_gradient = scalar_multiplication(hidden1_gradient, self.learning_rate)

                    inputs_t = transpose(inputs)
                    weight_ih_deltas = matrix_multiplication(hidden1_gradient, inputs_t)
                    
                    sample_weights_deltas.append(weight_ih_deltas)
                    sample_bias_deltas.append(hidden1_gradient)

                    weights_deltas.append(sample_weights_deltas)
                    bias_deltas.append(sample_bias_deltas)

                summed_weights_deltas = [[[0 for val in row] for row in layer] for layer in sample_weights_deltas]
                summed_bias_deltas = [[[0 for val in row] for row in layer] for layer in sample_bias_deltas]
                
                for sample in weights_deltas:
                    for j, layer in enumerate(sample):
                        for k, row in enumerate(layer):
                            for h, val in enumerate(row):
                                summed_weights_deltas[j][k][h] += val

                for sample in bias_deltas:
                    for j, layer in enumerate(sample):
                        for k, row in enumerate(layer):
                            for h, val in enumerate(row):
                                summed_bias_deltas[j][k][h] += val

                for layer in summed_weights_deltas:
                    for row in layer:
                        for val in row:
                            val /= batch_size

                for layer in summed_bias_deltas:
                    for row in layer:
                        for val in row:
                            val /= batch_size

                wd_matrix = summed_weights_deltas[0]
                bd_matrix = summed_bias_deltas[0]
                
                self.weights[-1] = elementwise_addition(self.weights[-1], wd_matrix)
                self.biases[-1] = elementwise_addition(self.biases[-1], bd_matrix)

                for i in range(self.hidden_layers - 1):
                    wd_matrix = summed_weights_deltas[i + 1]
                    bd_matrix = summed_bias_deltas[i + 1]
                    self.weights[-(i+2)] = elementwise_addition(self.weights[-(i+2)], wd_matrix)
                    self.biases[-(i+2)] = elementwise_addition(self.biases[-(i+2)], bd_matrix)

                wd_matrix = summed_weights_deltas[-1]
                bd_matrix = summed_bias_deltas[-1]
                self.weights[0] = elementwise_addition(self.weights[0], wd_matrix)
                self.biases[0] = elementwise_addition(self.biases[0], bd_matrix)

            print("Training")
            print("===================")
            if batch_size > 1:
                print("Epoch: ", e + 1)
                print("Mini-Batch Gradient Descent")
                print("Batch Size: ", batch_size)
                print("Loss: ", loss)
            elif batch_size == 1:
                print("Iteration: ", e + 1)
                print("Stochastic Gradient Descent")
                print('Loss: ', loss)
            print("===================")
            print('\n\n')

            losses.append(loss)

        if graph:
            NeuralNetwork.graph_loss(self, epoch_list, losses, batch_size)

    def train(self, input_array, targets_array, gd_type, epochs = 0, iterations = 0, batch_size = 8, graph = False):
        if 'batch' in gd_type.lower():
            NeuralNetwork.feed_and_propogate(self, input_array, targets_array, epochs, batch_size, graph)
        elif 'stochastic' in gd_type.lower():
            batch_size = 1
            NeuralNetwork.feed_and_propogate(self, input_array, targets_array, iterations, batch_size, graph)

    def graph_loss(self, epoch_list, loss_list, batch_size):
        pass

    def save_model(self):
        with open(r'C:\Users\auden\PycharmProjects\neural_network_lib\weights.pickle', 'wb') as file:
            pickle.dump((self.weights, self.biases), file)

    @staticmethod
    def load_model():
        with open(r'C:\Users\auden\PycharmProjects\neural_network_lib\weights.pickle', 'rb') as file:
            data = pickle.load(file)

        return data