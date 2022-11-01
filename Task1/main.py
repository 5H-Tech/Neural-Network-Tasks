import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('Datasets/penguins.csv')

def preprocessing(df):
    lab = LabelEncoder()
    # label encoding
    df['gender'] = lab.fit_transform(df['gender'])
    # fill missing values with median
    df["gender"] = df["gender"].replace(2, df["gender"].median())
    dfd = df.drop('species', inplace=False, axis=1)
    #feature scaleing
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(dfd), columns=dfd.columns)
    return df_scaled

dfr=preprocessing(df)

print(dfr)




