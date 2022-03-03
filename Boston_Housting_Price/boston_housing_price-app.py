from warnings import filterwarnings
from matplotlib.transforms import Bbox
filterwarnings('ignore')

import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False) # filter streamlit wargnins

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import pickle

st.write("""
# Boston House Price Prediction App

This app predicts the **Boston House Price**!
""")
st.write('---')

# Load the dataset
boston = datasets.load_boston()

X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.DataFrame(boston.target, columns=["MEDV"])


# Sidebar
# Input parameters
st.sidebar.header('Specify Input Parameters') 

def user_input_features():
    CRIM = st.sidebar.slider('CRIM', float(X.CRIM.min()),  float(X.CRIM.max()), float(X.CRIM.mean()))
    ZN = st.sidebar.slider('ZN', float(X.ZN.min()),  float(X.ZN.max()),  float(X.ZN.mean()))
    INDUS = st.sidebar.slider('INDUS', float(X.INDUS.min()),  float(X.INDUS.max()),  float(X.INDUS.mean()))
    CHAS = st.sidebar.slider('CHAS', float(X.CHAS.min()),  float(X.CHAS.max()),  float(X.CHAS.mean()))
    NOX = st.sidebar.slider('NOX', float(X.NOX.min()),  float(X.NOX.max()), float(X.NOX.mean()))
    RM = st.sidebar.slider('RM',float(X.RM.min()),  float(X.RM.max()),  float(X.RM.mean()))
    AGE = st.sidebar.slider('AGE',float(X.AGE.min()),  float(X.AGE.max()),  float(X.AGE.mean()))
    DIS = st.sidebar.slider('DIS', float(X.DIS.min()),  float(X.DIS.max()),  float(X.DIS.mean()))
    RAD = st.sidebar.slider('RAD', float(X.RAD.min()),  float(X.RAD.max()),  float(X.RAD.mean()))
    TAX = st.sidebar.slider('TAX', float(X.TAX.min()),  float(X.TAX.max()),  float(X.TAX.mean()))
    PTRATIO = st.sidebar.slider('PTRATIO', float(X.PTRATIO.min()),  float(X.PTRATIO.max()),  float(X.PTRATIO.mean()))
    B = st.sidebar.slider('B', float(X.B.min()),  float(X.B.max()),  float(X.B.mean()))	
    LSTAT = st.sidebar.slider('LSTAT', float(X.LSTAT.min()),  float(X.LSTAT.max()),  float(X.LSTAT.mean()))
    data = {
            'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,  
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE, 
            'DIS': DIS, 
            'RAD': RAD, 
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B':B,
            'LSTAT': LSTAT
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel
st.write('Specified input parameters')
st.write(df)
st.write('---')




# Load the dataset
boston = datasets.load_boston()

X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.DataFrame(boston.target, columns=["MEDV"])

# Load our model
load_model = pickle.load(open('boston_model.pkl', 'rb'))

# Apply saved model to make prediction 
st.subheader('Prediction of MEDV')
prediction = load_model.predict(df)
st.write(prediction)
st.write('---')

# Explaining the model prediction using SHAPE values
# summarize the effects of all the features
#https://shap.readthedocs.io/en/latest/index.html
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(load_model)
shap_values = explainer.shap_values(X)

st.header('Feature importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bb_inches='tight')