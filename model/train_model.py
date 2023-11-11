# Script to train machine learning model.
import os
import sys
import pickle
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from data import process_data
from model import train_model, compute_model_metrics, inference
sys.path.append(["../"])

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Add code to load in the data.
data = pd.read_csv("data/cleaned_census.csv")

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
    train, categorical_features=cat_features, label="salary", training=True)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(test, categorical_features=cat_features, 
                                    label="salary", training=False, 
                                    encoder=encoder, lb=lb)



# Train and save a model.
model = train_model(X_train, y_train)

save_path = './model/weight'
os.makedirs(save_path, exist_ok=True)

model_path = os.path.join(save_path, 'trainedmodel.pkl')
encoder_path = os.path.join(save_path, 'encoder.pkl')
lb_path = os.path.join(save_path, 'lb.pkl')

pickle.dump(model, open(model_path, 'wb'))
pickle.dump(encoder, open(encoder_path, 'wb'))
pickle.dump(lb, open(lb_path, 'wb'))

# Evaluate the model
y_preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_preds)

logger.info("Precision: %s", precision)
logger.info("Rrecall: %s", recall)
logger.info("Fbeta: %s", fbeta)
