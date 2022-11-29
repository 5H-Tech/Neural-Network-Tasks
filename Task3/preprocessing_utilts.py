import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


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