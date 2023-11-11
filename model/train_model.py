# Script to train machine learning model.
import os
import pickle
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from model.data import process_data
from model.model import train_model, compute_model_metrics, inference

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Add code to load in the data.
file_dir = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(file_dir,"./data/census_cleaned.csv"))

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
X_test, y_test, _, _ = process_data(test, categorical_features=cat_features, 
                                    label="salary", training=False, 
                                    encoder=encoder, lb=lb)



# Train and save a model.
model = train_model(X_train, y_train)

model_path = os.path.join(file_dir, './model/rf_model.pkl')
pickle.dump(model, open(model_path, 'wb'))

encoder_path = os.path.join(file_dir, './model/encoder.pkl')
pickle.dump(encoder, open(encoder_path, 'wb'))

lb_path = os.path.join(file_dir, './model/lb.pkl')
pickle.dump(lb, open(lb_path, 'wb'))

# Evaluate the model
y_preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_preds)

print("Precision: ", precision, " recall: ", recall, " fbeta: ", fbeta)

# Check slices
compute_slice_metrics(
        model=model,
        encoder=encoder,
        lb=lb,
        cleaned_df=data,
        categorical_features=cat_features,
        slice_features='education'
    )