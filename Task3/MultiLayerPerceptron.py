from Layer import Layer
from Task3.preprocessing_utilts import MNIST_preprocessing
from preprocessing_utilts import *
from matplotlib import pyplot as plt


class MultiLayerPerceptron:
    def __init__(self, learning_rate=0.1, epics=100, is_sigmoid=True):
        self.layers = []
        self.learning_rate = learning_rate
        self.epics = epics
        self.is_sigmoid = is_sigmoid
        self.accuracy_list = []
        self.ep_list = []

    def add_output_layer(self, no_neurons):
        self.layers.append(Layer(no_neurons, is_output_layer=True, is_sigmoid=self.is_sigmoid))

    def add_hidden_layer(self, no_neurons):
        tmp = Layer(no_neurons, is_output_layer=False, is_sigmoid=self.is_sigmoid)
        tmp.attach(layer=self.layers[0])
        self.layers.insert(0, tmp)

    def update_layers(self, target):

        for layer in reversed(self.layers):
            for neuron in layer.neurons:
                neuron.clc_update(target)

        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.update_weights(learning_rete=self.learning_rate)

    def fit(self, x, y):
        rows = x.shape[0]
        num_festers = x.shape[1]

        self.layers[0].init(num_festers)

        for i in range(1, len(self.layers)):
            num_input = len(self.layers[i - 1].neurons)
            self.layers[i].init(num_input)

        for i in range(self.epics):
            for j in range(rows):
                row = x[j]  # take the random sample from the dataset
                yhat = self.predict(row)
                target = y[j]

                self.update_layers(target)

            # if i % 100 == 0:
            acc = self.predict_and_get_accuracy(x, y, 'Train', i)
            self.accuracy_list.append(acc)
            self.ep_list.append(i)
            if acc >= 1:
                break

    def predict_and_get_accuracy(self, x, y, type="Train", epoc=-1):
        num_rows = x.shape[0]
        right = 0
        for t_i in range(num_rows):
            y_hat = self.predict(x[t_i])
            # print(y_hat)
            # print(y[t_i])
            a = True
            for k in range(y.shape[1]):
                if y[t_i][k] != y_hat[k]:
                    a = False
                    break
            if a:
                right += 1
            acc = right / y.shape[0]
        if epoc == -1:
            print(f'{type} accuracy ={acc}')
        else:
            print(f'at epochs number {epoc} accuracy =={acc}')
        return acc

    def predict(self, row):
        activations = self.layers[0].predict(row)
        for i in range(1, len(self.layers)):
            activations = self.layers[i].predict(activations)

        outputs = []
        max_val = max(activations)

        for activation in activations:
            if activation == max_val:
                outputs.append(1.0)
            else:
                outputs.append(0.0)

        # We currently have only One output allowed
        return outputs

    def plot_accuracy_graph(self):
        plt.plot(self.ep_list, self.accuracy_list)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()

# Multilayer testing
if __name__ == "__main__":
    x_train, x_test, y_train, y_test = preprocessing()
    clf = MultiLayerPerceptron(0.3, 10000, True)
    clf.add_output_layer(3)
    clf.add_hidden_layer(2)
    clf.add_hidden_layer(3)
    clf.add_hidden_layer(4)
    clf.fit(x_train, y_train)
    clf.predict_and_get_accuracy(x_test, y_test, "Test")

