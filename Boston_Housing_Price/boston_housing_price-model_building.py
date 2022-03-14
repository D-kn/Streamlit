# Import librairies
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load the dataset
boston = datasets.load_boston()

X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.DataFrame(boston.target, columns=["MEDV"])


# Build Regression model
model = RandomForestRegressor()
model.fit(X, y)

# Save our model
pickle.dump(model, open('boston_model.pkl', 'wb'))


