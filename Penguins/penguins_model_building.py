import numpy as np
import pandas as pd
import seaborn as sns
 

penguins = pd.read_csv('./penguins_dataset.csv')

df = penguins.copy()

target = df['species']
encode = ['sex', 'island']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

target_mapper = {'Adelie':0, 'Gentoo':1, 'Chinstrap':2}
def target_encode(val):
    return target_mapper[val]


df['species'] = df['species'].apply(target_encode)


# Defining X and y variables
X = df.drop('species', axis=1)
y = df['species']


# Model Building : Random Forest
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X, y)


# Save the model
import pickle

pickle.dump(model, open('penguins_model.pkl', 'wb'))