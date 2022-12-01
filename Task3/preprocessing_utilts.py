import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from os import path


def preprocessing():
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
    return x_train, x_test, y_train, y_test


def MNIST_preprocessing():
    x_train = None
    x_test = None
    y_train =None
    y_test = None
    if path.exists("x_train.npy") \
            and path.exists("x_test.npy") \
            and path.exists("y_train.npy") \
            and path.exists("y_test.npy"):
        x_train = np.load('x_train.npy')
        x_test = np.load('x_test.npy')
        y_train = np.load('y_train.npy')
        y_test = np.load('y_test.npy')
    else:
        df = pd.read_csv('Datasets/MINIST/mnist_train.csv')
        y = pd.get_dummies(df.label, prefix='')
        y = np.array(y)
        df = df.drop('label', axis=1)
        scaler = MinMaxScaler()
        for col in df.columns.values:
            df[col] = scaler.fit_transform(df[[col]])
        x = np.array(df)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=42)
        np.save('x_train.npy', x_train)
        np.save('x_test.npy', x_test)
        np.save('y_train.npy', y_train)
        np.save('y_test.npy', y_test)
    return x_train, x_test, y_train, y_test
