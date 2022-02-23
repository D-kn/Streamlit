import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sb 
from sklearn.ensemble import RandomForestClassifier



st.write("""
# Simple Iris Flower Prediction App

This app predicts the **Iris flower** type!
""")

st.sidebar.header('USer Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.8)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.10)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 3.50)
    petal_width = st.sidebar.slider('Petal width', 0.1,  2.5, 1.10)

    data = {'sepal length': sepal_length,
            'sepal width': sepal_width,
            'petal length': petal_length,
            'petal width': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

iris = sb.load_dataset('iris')
# print(iris.shape)
X = iris.drop(['species'], axis=1)
y = iris['species']

# print(X)
# print(y)

model = RandomForestClassifier()
model.fit(X, y)

prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

print(prediction)

st.subheader('Class labels and their corresponding index number')
st.write(iris.species.unique())

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)












































