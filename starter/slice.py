from ml.model import *
from ml.data import *
import json

def performance_feature(test,encoder,lb, model, cat_features):

    results = {}
    for feature in cat_features:
        performance = {}
        for label in test[feature].unique():
            data = test[test[feature] == label]
            X, y_test, encoder, lb = process_data(
            data, categorical_features=cat_features, label="salary", training=False,
            encoder = encoder, lb = lb)
            preds = inference(model, X)
            precision, recall, fbeta = compute_model_metrics(y_test, preds)
            performance[label] = {'precision':precision,'recall':recall,'fbeta':fbeta}
        results[feature] = performance
        
    
    with open('../screenshots/slice_output.txt', 'w') as file:
        file.write(json.dumps(results)) # use `json.loads` to do the reverse



