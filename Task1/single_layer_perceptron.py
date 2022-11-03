from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
pd.options.mode.chained_assignment = None


class slp:
    def __init__(self, learning_rate=0.01, epchos=1000, is_bise=1,
                 first_featur='', second_featuer='',
                 first_class='', second_class=''):
        self.lr = learning_rate
        self.epchos = epchos
        self.is_bise = is_bise
        self.activation_func = self._unit_step_func
        self.weights = np.zeros(2)
        self.bias = np.random.rand()
        self.first_featur = first_featur
        self.second_featur = second_featuer
        self.first_class = first_class
        self.second_class = second_class
        self.lable = 'species'
        self.dataset = None
        self.X = None
        self.Y = None
        self.x_test = None
        self.y_test = None
        self.x_tran = None
        self.y_tran = None

    def preprocessing(self, df=pd.read_csv('Datasets/penguins.csv')):
        print('preprocessing started ....')
        lab = LabelEncoder()
        # label encoding
        df['gender'] = lab.fit_transform(df['gender'])
        df['gender'] = df['gender'].replace(2, df['gender'].median())
        # drop not needed data
        for col in df.columns.values:
            if col != self.first_featur and col != self.second_featur and col != self.lable:
                df.drop(col, axis=1, inplace=True)
        types = ['Chinstrap','Adelie','Gentoo']
        types.remove(self.first_class)
        types.remove(self.second_class)
        df = df[df.species != types[0]]
        df[self.lable] = df[self.lable].replace(self.first_class, -1)
        df[self.lable] = df[self.lable].replace(self.second_class, 1)
        self.dataset = df
        self.X = np.array(self.dataset[[self.first_featur, self.second_featur]])
        self.Y = np.array(self.dataset[self.lable])

        self.x_tran, self.x_test, self.y_tran, self.y_test = train_test_split(self.X, self.Y, test_size=0.1,
                                                                              shuffle=True,
                                                                              random_state=123)
        print('preprocessing done!')

    def train_phase(self):
        new_weights = self.weights
        new_bias = self.bias * self.is_bise
        # Looping over all samples
        # for i in range(len(first_feature)):
        #     value = (first_feature.values[i] * new_weight1) + (second_feature.valuse[i] * new_weight2) + (
        #             is_bias * new_bias)
        #     # Signum Activation function
        #     if value < 0:
        #         y = -1
        #     else:
        #         y = 1
        #     error = target_output[i] - y
        #     # Updating the weights and bias
        #     new_weight1 += (learning_rate * first_feature[i] * error)
        #     new_weight2 += (learning_rate * second_feature[i] * error)
        #     new_bias += (learning_rate * is_bias * error)
        for idx, x_i in enumerate(self.x_tran):
            net_output = np.dot(x_i, new_weights) + new_bias
            y_hat = 1 if net_output >= 0 else -1

            updata = self.lr * (self.y_tran[idx] - y_hat)
            print(f'the weights is updated by : {updata}')
            new_weights += updata * x_i
            new_bias = updata

        return new_weights, new_bias

    def run_model(self):
        # Loops the training process depending on the epochs
        for i in range(self.epchos):
            self.weights, self.bias = self.train_phase()

    # def fit(self, X, y):
    #     n_samples, n_features = X.shape
    #
    #     # init parameters
    #     self.weights = np.zeros(n_features)
    #     self.bias = 0
    #
    #     # y_ = np.array([1 if i > 0 else -1 for i in y])
    #
    #     for _ in range(self.epchos):
    #
    #         for idx, x_i in enumerate(X):
    #             linear_output = np.dot(x_i, self.weights) + self.bias
    #             y_predicted = self.activation_func(linear_output)
    #
    #             # Perceptron update rule
    #             update = self.lr * (y[idx] - y_predicted)
    #
    #             self.weights += update * x_i
    #             self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, -1)

    def accuracy(self, y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    def evalution(self, y_pred_test):
        print("Perceptron classification accuracy", self.accuracy(self.y_test, y_pred_test))

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(self.x_test[:, 0], self.x_test[:, 1], marker="o", c=self.y_test)
        plt.title(f'{self.first_class} vs {self.second_class} ')
        plt.xlabel(self.first_featur)
        plt.ylabel(self.second_featur)
        x0_1 = np.amin(self.x_test[:, 0])
        x0_2 = np.amax(self.x_test[:, 0])

        x1_1 = (-self.weights[0] * x0_1 - self.bias) / self.weights[1]
        x1_2 = (-self.weights[0] * x0_2 - self.bias) / self.weights[1]

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

        ymin = np.amin(self.x_test[:, 1])
        ymax = np.amax(self.x_test[:, 1])
        ax.set_ylim([ymin - 3, ymax + 3])
        plt.show()

    # Bulid Confusion Matrix using actual and predicted data
    def confusion_matrix(self,actual_data, predicted_data):
        # Create a Zip which is an iterator of tuples that returns each item in the list with its counterpart in the other list
        key = zip(actual_data, predicted_data)
        dict = {}

        # Loop to add tuple as key in dictionay and update The number of times it appears
        for actual, predicted in key:
            if (actual, predicted) in dict:
                dict[(actual, predicted)] += 1
            else:
                dict[(actual, predicted)] = 1

        # Convert Dictionary to Series
        sr = pd.Series(list(dict.values()), index=pd.MultiIndex.from_tuples(dict.keys()))
        # Convert Series to Dataframe
        df = sr.unstack().fillna(0)
        return df

    # Plot Confusion Matrix as heatmap
    def plot_confusion_matrix(self,actual_list, predicted_list):
        con_mat = self.confusion_matrix(actual_list, predicted_list)
        hm = sns.heatmap(con_mat, annot=True, xticklabels=con_mat.index, yticklabels=con_mat.columns)
        hm.invert_yaxis()
        plt.xlabel("Predicted Values")
        plt.ylabel("Actual Values")
        plt.show()
        return True

# if __name__ == "__main__":
#     # SLP = slp(first_class='Adelie', second_class='Gentoo', first_featur='gender', second_featuer='bill_length_mm')
#     # res = SLP.preprocessing()
#     # X = res[[SLP.first_featur, SLP.second_featur]]
#     # X = np.array(X)
#     # Y = res[SLP.lable]
#     # Y = np.array(Y)
#     # x_tran, y_tran, x_test, y_test = train_test_split(X, Y, test_size=0.4, shuffle=True, random_state=123)
#     # SLP.fit(x_tran, y_tran)
#     # y_hat = SLP.predict(x_test, y_test)
