import pickle
import pandas as pd
from build_model import TextClassifier, get_data

with open('static/model.pkl', 'rb') as f:
    model = pickle.load(f)

X, y = get_data('data/articles.csv')

print("Accuracy:", model.score(X, y))
print("Predictions:", model.predict(X))
print("Predict one:", model.predict(["You should save the model as a pickle file. This is python's internal format for saving objects to disk. Almost all python objects can be pickled, and then reloaded in another python process. Note though, to reload a pickled object, you must import the class defining that object in the process that would like to do the loading."]))
