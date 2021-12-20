from ml.model import *
from ml.data import *
import json

def performance_feature(test,encoder,lb, model, cat_features):

    performance = {"feature":[],"precision":[],"recall":[],"fbeta":[]}
    for feature in cat_features:
        X = test[feature]
        X, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False,
        encoder = encoder, lb = lb)
        preds = inference(model, X)
        precision, recall, fbeta = compute_model_metrics(y_test, preds)
        performance['feature'].append(feature)
        performance['precision'].append(precision)
        performance['recall'].append(recall)
        performance['fbeta'].append(fbeta)
    
    with open('../screenshots/slice_outouts.txt', 'w') as file:
        file.write(json.dumps(performance)) # use `json.loads` to do the reverse



