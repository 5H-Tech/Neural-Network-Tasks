import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def lableEncoder(df, column_name, label_list):
    encoded_column = df[column_name].replace(label_list)
    return encoded_column

def get_missing_value(df,column):
    model = LinearRegression()
    data = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'gender']]
    data_without_missing = data.dropna()
    train_x = data_without_missing.iloc[:, :4]
    train_y = data_without_missing.iloc[:, 4]
    model.fit(train_x, train_y)
    test_x = data.iloc[:, :4]
    result = np.round(np.abs(model.predict(test_x)))
    df[column].fillna(pd.Series(result), inplace=True)
    return df

def preprocessing_data(df):
    list1 = {"male": 0, "female": 1}
    df['gender'] = lableEncoder(df, "gender", list1)

    list2 = {"Adelie": 1, "Gentoo": 2,"Chinstrap": 3}
    df['species'] = lableEncoder(df, "species", list2)
    df = get_missing_value(df,"gender")
    return df
def train():
    return
def test():
    return

df = pd.read_csv("Datasets/penguins.csv")
df = preprocessing_data(df)
sns.pairplot(df, hue="species",height=1.5)
plt.show()







