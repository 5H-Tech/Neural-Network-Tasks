import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

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
    for i in range(len(first_feature)):
        value = (first_feature[i] * weight1) + (second_feature[i] * weight2) + (is_bias * bias)
        # Signum Activation function
        if(value < 0):
            y = -1
        else:
            y = 1
        error = target_output[i] - y
        #weight1 = weight1 + (learning_rate * first_feature[i] * error)
        #weight2 = weight2 + (learning_rate * second_feature[i] * error)
        #bias = bias + (learning_rate * is_bias * error)

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

    # This is just a placeholder for the data
    #-----------------------------------------------------Split the data (60% train, 40% test)--------------------------------------------------
    first_feature_gui = dfr[first_feature]
    second_feature_gui = dfr[second_feature]
    target_output_gui = dfr['species']

    # Loops the training process depending on the epochs
    for i in range(epochs):
        train_phase(first_feature_gui, second_feature_gui, target_output_gui, is_bias, learning_rate)
    
