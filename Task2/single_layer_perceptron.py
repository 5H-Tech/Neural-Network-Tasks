from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

pd.options.mode.chained_assignment = None


def linear_func(x):
    return np.where(x >= 0, 1, -1)


def get_accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


class Slp:
    def __init__(self, learning_rate=0.01, epochs=1000, is_bias=1,
                 first_feature='', second_feature='',
                 first_class='', second_class='', threshold=0.2):
        self.lr = learning_rate
        self.epochs = epochs
        self.is_bias = is_bias
        self.mapping_func = linear_func
        self.weights = np.random.rand(2)
        self.bias = np.random.rand()
        self.first_feature = first_feature
        self.second_feature = second_feature
        self.first_class = first_class
        self.second_class = second_class
        self.mse_threshold = threshold
        self.label = 'species'
        self.x_test = None
        self.y_test = None
        self.x_train = None
        self.y_train = None

    def preprocessing(self, df=pd.read_csv('Datasets/penguins.csv')):
        print('preprocessing started ....')
        lab = LabelEncoder()
        # label encoding
        df['gender'] = lab.fit_transform(df['gender'])
        df['gender'] = df['gender'].replace(2, df['gender'].median())
        scaler = MinMaxScaler()

        df[[self.first_feature]] = scaler.fit_transform(df[[self.first_feature]])
        df[[self.second_feature]] = scaler.fit_transform(df[[self.second_feature]])
        self.three_classes_evaluation(df)
        # drop unnecessary data
        for col in df.columns.values:
            if col != self.first_feature and col != self.second_feature and col != self.label:
                df.drop(col, axis=1, inplace=True)
        types = ['Chinstrap', 'Adelie', 'Gentoo']
        types.remove(self.first_class)
        types.remove(self.second_class)
        df = df[df.species != types[0]]
        df[self.label] = df[self.label].replace(self.first_class, -1)
        df[self.label] = df[self.label].replace(self.second_class, 1)

        first_train, first_test = train_test_split(df[df[self.label] == -1], test_size=0.4, train_size=0.6,
                                                   shuffle=True)
        second_train, second_test = train_test_split(df[df[self.label] == 1], test_size=0.4, train_size=0.6,
                                                     shuffle=True)

        train_data = pd.concat([first_train, second_train], ignore_index=True).sample(frac=1).reset_index(drop=True)
        test_data = pd.concat([first_test, second_test], ignore_index=True).sample(frac=1).reset_index(drop=True)

        self.x_train = np.array(train_data[[self.first_feature, self.second_feature]])
        self.y_train = np.array(train_data[self.label])
        self.x_test = np.array(test_data[[self.first_feature, self.second_feature]])
        self.y_test = np.array(test_data[self.label])
        print('preprocessing done!')

    def train_phase(self):
        new_weights = self.weights
        new_bias = self.bias
        error_summation = 0

        for idx, x_i in enumerate(self.x_train):
            net_output = np.dot(x_i, new_weights) + new_bias
            y_hat = net_output
            error = self.y_train[idx] - y_hat
            error_summation += error*error*1/2
            new_weights += self.lr * error * x_i
            new_bias += self.lr * error*self.is_bias
        return new_weights, new_bias, error_summation

    def run_model(self):
        # Loops the training process till reaching the threshold
        for i in range(10000):
            self.weights, self.bias, error_summation = self.train_phase()
            mse = error_summation/self.x_train.size
            if mse <= self.mse_threshold:
                break

    def predict(self, x):
        linear_output = np.dot(x, self.weights) + self.bias
        y_predicted = self.mapping_func(linear_output)
        return y_predicted

    def evaluation(self, y_pred_test):
        print("Perceptron classification accuracy", get_accuracy(self.y_test, y_pred_test))

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.title(f'{self.first_class} vs {self.second_class} ')
        plt.xlabel(self.first_feature)
        plt.ylabel(self.second_feature)
        plt.scatter(self.x_test[:, 0], self.x_test[:, 1], marker="o", c=self.y_test)

        x0_1 = np.amin(self.x_test[:, 0])
        x0_2 = np.amax(self.x_test[:, 0])

        x1_1 = (-self.weights[0] * x0_1 - self.bias) / self.weights[1]
        x1_2 = (-self.weights[0] * x0_2 - self.bias) / self.weights[1]

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

        ymin = np.amin(self.x_test[:, 1])
        ymax = np.amax(self.x_test[:, 1])
        ax.set_ylim([ymin - 0.1, ymax + 0.1])
        plt.show()


    def three_classes_evaluation(self,df):
        sns.pairplot(df, hue="species", height=2,corner=True)
        plt.show()

    # Build Confusion Matrix using actual and predicted data
    def confusion_matrix(self, Actual_data, Predicted_data):
        # Create a Zip which is an iterator of tuples that returns each item in the list with its counterpart
        # in the other list
        actual_data = [self.first_class if x==1 else self.second_class for x in Actual_data]
        predicted_data = [self.first_class if x==1 else self.second_class for x in Predicted_data]
        key = zip(actual_data, predicted_data)
        dict = {}

        # Loop to add tuple as key in dictionary and update The number of times it appears
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
    def plot_confusion_matrix(self, actual_list, predicted_list):
        con_mat = self.confusion_matrix(actual_list, predicted_list)
        hm = sns.heatmap(con_mat, annot=True, xticklabels=con_mat.index, yticklabels=con_mat.columns)
        hm.invert_yaxis()
        plt.xlabel("Predicted Values")
        plt.ylabel("Actual Values")
        plt.show()
        return True

