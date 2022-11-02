import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('Datasets/penguins.csv')
weight1 = np.random.rand()
weight2 = np.random.rand()
bias = np.random.rand()
species_label_dictionary = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}

def preprocessing(df):
    lab = LabelEncoder()
    # label encoding
    df['gender'] = lab.fit_transform(df['gender'])
    df['species'] = lab.fit_transform(df['species'])
    # fill missing values with median
    df["gender"] = df["gender"].replace(2, df["gender"].median())
    #feature scaling
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    # This line is to return the label encoding to normal (It gets normalized from scaler.fit_transform above this line)
    df_scaled['species'] = lab.fit_transform(df_scaled['species']) 
    return df_scaled

def train_phase(first_feature, second_feature, target_output, is_bias, learning_rate):
    new_weight1 = weight1
    new_weight2 = weight2
    new_bias = bias
    # Looping over all samples
    for i in range(len(first_feature)):
        value = (first_feature[i] * new_weight1) + (second_feature[i] * new_weight2) + (is_bias * new_bias)
        # Signum Activation function
        if(value < 0):
            y = -1
        else:
            y = 1
        error = target_output[i] - y
        # Updating the weights and bias
        new_weight1 += (learning_rate * first_feature[i] * error)
        new_weight2 += (learning_rate * second_feature[i] * error)
        new_bias += (learning_rate * is_bias * error)
    
    return new_weight1, new_weight2, new_bias

# Made in a function to run it directly from the window.py file (Implemented)
def run_model(first_feature, second_feature, first_class, second_class, is_bias, epochs, learning_rate):
    # Start of the preprocessing
    dfr=preprocessing(df)
    dfr.to_csv("Datasets/updated_data_penguins.csv")

    #Removes the unchosen class from the dataframe
    first_class_number = species_label_dictionary[first_class]
    second_class_number = species_label_dictionary[second_class]
    dropped_class = sum(list(dfr["species"].unique())) - first_class_number - second_class_number
    dfr = dfr[dfr['species'] != dropped_class]

    dfr['species'] = dfr['species'].replace(first_class_number, -1)
    dfr['species'] = dfr['species'].replace(second_class_number, 1)

    # This is just a placeholder for the data
    #-----------------------------------------------------Split the data (60% train, 40% test)--------------------------------------------------
    first_train, first_test = train_test_split(dfr[dfr['species'] == -1], test_size=0.4, train_size=0.6, shuffle=True)
    second_train, second_test = train_test_split(dfr[dfr['species'] == 1], test_size=0.4, train_size=0.6, shuffle=True)
    merged_train = first_train
    merged_test = first_test

    # Loops the training process depending on the epochs
    for i in range(epochs):
        weight1, weight2, bias = train_phase(merged_train[first_feature], merged_train[second_feature], merged_train['species'], is_bias, learning_rate)
    
