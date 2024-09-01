import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.decomposition import PCA

def load_data(df):
    #remove non numeric rows
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    df_numeric.dropna(inplace = True , axis = 0)
    return df_numeric

def scaling(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled

def model(df):
    model = load_model('/Users/aditya/Desktop/assignments/Autoencoder.keras')
    preds = model.predict(df)
    preds[preds > 0.5] = 1
    preds[preds < 0.5] = 0
    return preds

def plot(preds , df):
    composer = PCA(n_components=2)
    transformed = composer.fit_transform(df)
    sns.scatterplot(x = transformed[:,0] , y = transformed[:,1] , hue = encoder_preds)
    plt.title('AutoEncoder Predicted')
    plt.savefig('anomalies_plot.png')