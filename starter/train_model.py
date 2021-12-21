# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.model import *
from ml.data import *
from slice import *
# Add the necessary imports for the starter code.
import pandas as pd
import joblib
# Add code to load in the data.
data = pd.read_csv("../data/census.csv")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder = encoder, lb = lb
)
# Train and save a model.
model = train_model(X_train, y_train)


y_preds = inference(model, X_test)

print(compute_model_metrics(y_test, y_preds))


performance_feature(test,encoder,lb,model,cat_features)

joblib.dump(model, "../model/model.pkl")
joblib.dump(encoder, "../model/encoder.pkl")
joblib.dump(lb, "../model/lb.pkl")