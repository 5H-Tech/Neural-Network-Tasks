import numpy as np
from preprocessing_utilts import preprocessing


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    ds = s * (1 - s)
    return s, ds


def tanh(x):
    t = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    dt = 1 - t ** 2
    return t, dt


class Neuron:

    def __init__(self, layer_pos=0, is_output=False, is_sigmoid=True):
        self.inputs = None
        self.is_output = is_output
        self.is_sigmoid = is_sigmoid
        self.weights = None
        self.layer_pos = layer_pos
        self.next_neurons = None
        self.output = 0
        self.d_output = 0
        self.local_error = 0
        self.der_local_error = 0

    def init(self, input_size):
        self.weights = np.random.rand(input_size + 1)

    def attatch(self, neurons):
        self.next_neurons = neurons

    # forward step
    def predict(self, input):
        self.inputs = np.append(input, 1)
        tmp_output = np.dot(self.inputs, self.weights)
        self.output, self.d_output = sigmoid(tmp_output) if self.is_sigmoid else tanh(tmp_output)
        return self.output

    # backward step
    def clc_update(self, target):
        if self.is_output:
            self.local_error = (target[self.layer_pos] - self.output) * self.d_output
        else:
            sum = 0
            for nur in self.next_neurons:
                sum += nur.weights[self.layer_pos] * nur.local_error
            self.local_error = sum * self.d_output

    # forward step
    def update_weights(self, learning_rete):
        self.weights += self.inputs * self.local_error * learning_rete


# testing the neuron
if __name__ == "__main__":
    x_train, x_test, y_train, y_test = preprocessing()
    #
    n1 = Neuron(1, False, True)  # input
    n2 = Neuron(1, True, True)  # output
    n1.init(x_test.shape[1])
    n2.init(x_test.shape[1])
    a = [n2]
    n1.attatch(a)

    # foreword
    n1.predict(x_train[1])
    n2.predict(x_train[0])
    # BACKWARD
    n2.clc_update(y_test[-1])
    n1.clc_update(y_test[0])
    # FOREWORD
    n1.update_weights(0.1)
    n2.update_weights(0.1)
