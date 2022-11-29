from Neuron import Neuron
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class Layer:
    def __init__(self, number_neurons, is_output_layer=False, is_sigmoid=True):
        self.is_output_layer = is_output_layer
        self.is_sigmoid = is_sigmoid
        self.neurons = []
        for i in range(number_neurons):
            n = Neuron(i, self.is_output_layer, self.is_sigmoid)
            self.neurons.append(n)

    def init(self, input_size):
        for i in range(len(self.neurons)):
            self.neurons[i].init(input_size)

    def attach(self, layer):
        for i in range(len(self.neurons)):
            self.neurons[i].attatch(layer.neurons)

    def predict(self, input):
        activations = [neuron.predict(input) for neuron in self.neurons]
        return activations


# testing the layer
if __name__ == "__main__":
    print('preprocessing started ....')
    df = pd.read_csv('Datasets/penguins.csv')
    lab = LabelEncoder()
    # label encoding
    df['gender'] = lab.fit_transform(df['gender'])
    df['gender'] = df['gender'].replace(2, df['gender'].median())
    scaler = MinMaxScaler()
    y = pd.get_dummies(df.species, prefix='')
    df.drop('species', axis=1, inplace=True)
    for col in df.columns.values:
        df[col] = scaler.fit_transform(df[[col]])
    x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print('preprocessing done!')

    l1 = Layer(3, False, False)  # hidden 1
    l2 = Layer(4, False, False)  # hidden 2
    l3 = Layer(3, True, False)  # Output

    # init
    l1.init(5)
    l2.init(3)
    l3.init(4)

    l1.attach(l2)
    l2.attach(l3)

    for j in range(1000):
        tmp = []
        for i in range(x_train.shape[0]):
            tmp = []
            # moving forward
            l1_output = l1.predict(x_train[i])
            l2_output = l2.predict(l1_output)
            l3_output = l3.predict(l2_output)
            # print(l3_output)

            for out in l3_output:
                if out > 0.5:
                    tmp.append(1)
                else:
                    tmp.append(0)
            # print(tmp)
            # calculate the local error (moving backworde)
            for n in l3.neurons:
                n.clc_update(y_train[i])  # will be used here
                # print(n.weights)
            for n in l2.neurons:
                n.clc_update(y_train[i])  # will not be used here
                # print(n.weights)
            for n in l1.neurons:
                n.clc_update(y_train[i])  # will not be used here
                # print(n.weights)
            # updating the weights (moving forward again)

            # print(" -===========--------======")
            for n in l1.neurons:
                n.update_weights(0.1)
                # print(n.weights)
            for n in l2.neurons:
                n.update_weights(0.1)
                # print(n.weights)
            for n in l3.neurons:
                n.update_weights(0.1)
                # print(n.weights)
        # print(tmp)
    right = 0
    for i in range(x_test.shape[0]):
        tmp = []
        l1_output = l1.predict(x_test[i])
        l2_output = l2.predict(l1_output)
        l3_output = l3.predict(l2_output)
        for out in l3_output:
            if out > 0.5:
                tmp.append(1)
            else:
                tmp.append(0)
        # print(tmp)
        # print(f'the real is {y_test[i]} predected is {tmp}')

        a = True
        for k in range(3):
            if y_test[i][k] != tmp[k]:
                a = False
        if a:
            right += 1
    print(f'accrucy ={right / x_test.shape[0]}')
