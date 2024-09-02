import streamlit as st
import pandas as pd
from models import *
from PIL import Image
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://images.pexels.com/photos/259200/pexels-photo-259200.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2");
    background-size: cover;
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)

def main():
    st.title("Credit Card Transaction Anomaly Detection")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        scaled = scaling(data)
        preds = model(scaled)
        plot(preds , scaled)
        st.image('anomalies_plot.png', caption='Anomaly Detection Plot')
        st.write(f"detected {len(preds[preds == 1])} Credit Card Frauds")

if __name__ == "__main__":
    main()
