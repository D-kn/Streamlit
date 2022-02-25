from distutils.command.upload import upload
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import pickle
# from penguins_model_building import *

st.write("""
# Penguin Prediction App

This app predicts the **Palmer Penguin** species!
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
# [Example CSV input file](https://github.com/D-kn)
""")

# User input into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["CSV"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island', ('Torgersen', 'Biscoe', 'Dream'))
        sex = st.sidebar.selectbox('Sex', ('female', 'male'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1, 59.6, 44.5)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1, 21.5, 17.10)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0, 231.0, 200.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0, 6300.0, 4050.0)

        data = {
            'island':island, 
            'sex': sex,
            'bill_length_mm': bill_length_mm, 
            'bill_depth_mm': bill_depth_mm, 
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g
        }

        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

# combine user input features with entire penguis dataset
penguins_raw = pd.read_csv('penguins_dataset.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df, penguins], axis=0)

encode = ['sex', 'island']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1]
# print(df)


st.subheader('User Input Features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below)')
    st.write(df)


# Read in our saved classification model 
load_model = pickle.load(open('penguins_model.pkl', 'rb'))


#Apply model to make prediction
prediction = load_model.predict(df)
prediction_proba = load_model.predict_proba(df)

st.subheader('Prediction')
penguins_species = np.array(['Adelie', 'Gentoo', 'Chinstrap'])
st.write(penguins_species[prediction])

st.subheader('Class labels and their corresponding index number')
st.write(penguins_raw.species.unique())

st.subheader('Prediction Probability')
st.write(prediction_proba)