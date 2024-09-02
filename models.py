import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.decomposition import PCA
import requests

model_url = 'https://github.com/dark-horiznz/Assignment/blob/3c556c142996089c55e80da52dd3dff73e3a404c/Autoencoder.keras'
local_path = 'my_model.keras'
response = requests.get(model_url)
if response.status_code == 200:
    with open(local_path, 'wb') as file:
        file.write(response.content)
else:
    print(f"Failed to download file from {url}. Status code: {response.status_code}")

def load_data(df):
    #remove non numeric rows
    df = pd.read_csv(df)
    for col in df.columns:
        for i in df[col]:
            if type(i) == str:
                try:
                    i = float(i)
                except:  
                    print(i)
                    df = df.drop(df[df[col] == i].index)
    return df

def scaling(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled

def model(df):
    model = load_model('my_model.keras')
    preds = model.predict(df)
    preds[preds > 0.5] = 1
    preds[preds < 0.5] = 0
    return preds

def plot(preds , df):
    composer = PCA(n_components=2)
    transformed = composer.fit_transform(df)
    sns.scatterplot(x = transformed[:,0] , y = transformed[:,1] , hue = preds.flatten())
    plt.title('AutoEncoder Predicted')
    plt.savefig('anomalies_plot.png')
