# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

    Author: Edgar Arellano
    Model : It is a GradientBoostingClassifier using the default hyperparameters in scikit-learn

## Intended Use

    Predict if a individual earns more than $50k based on census data

## Training Data

    The model was trainen using the 80% of the dataset

## Evaluation Data

    The rest 20% was used to evaluate the model

## Metrics
_Please include the metrics used and your model's performance on those metrics._
 
 precision = 0.7910321489001692
 recall =  0.5982085732565579
 fbeta =  0.6812386156648451

## Ethical Considerations
 
    NA
   
## Caveats and Recommendations

    Using a bigest dataset to improve the model training
