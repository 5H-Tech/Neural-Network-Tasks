import numpy as np
import random
import math


class Neuron:

    def __init__(self, position_in_layer, is_sigmoid=True, is_output_neuron=False):
        self.output_neurons = None
        self.weights = None
        self.inputs = None
        self.output = None
        self.doutput = None
        self.is_sigmoid = is_sigmoid

        # This is used for the backpropagation update
        self.updated_weights = None
        # This is used to know how to update the weights
        self.is_output_neuron = is_output_neuron
        # This delta is used for the update at the backpropagation
        self.delta = None
        # This is used for the backpropagation update
        self.position_in_layer = position_in_layer

    def attach_to_output(self, neurons):
        self.output_neurons = neurons

    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        ds = s * (1 - s)
        return s, ds

    def tanh(self, x):
        t = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        dt = 1 - t ** 2
        return t, dt

    def init_weights(self, num_input):
        # the number of input and basi
        self.weights = np.random.rand(num_input + 1)

    def predict(self, row):
        self.inputs = np.insert(row, 0, 1)

        net = np.dot(self.inputs, self.weights)
        self.output, self.doutput = self.sigmoid(net) if self.is_sigmoid else self.tanh(net)

        return self.output

    def update_neuron(self):

        self.weights = self.updated_weights

    def calculate_update(self, learning_rate, target):

        if self.is_output_neuron:
            # Calculate the delta for the output
            self.delta = (self.output - target) * self.doutput
        else:
            # Calculate the delta
            delta_sum = 0
            for output_neuron in self.output_neurons:
                delta_sum = delta_sum + (output_neuron.delta * output_neuron.weights[self.position_in_layer])

            # Update this neuron delta
            self.delta = delta_sum * self.doutput

        self.updated_weights = self.weights + learning_rate * self.delta * self.inputs