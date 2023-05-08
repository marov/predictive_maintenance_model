import pandas as pd
import streamlit as st
from ml_model.ml_xgboost import XGBClassifierModel
from predictive_maintenance.modeler import PredictiveMaintananceModeler

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

modeler = PredictiveMaintananceModeler()
modeler.load_data() # in "real life" this would be loading from IoT sensor ETL pipeline
model = XGBClassifierModel()
model.load('xgb_classifier_1st_version.pkl')
inference_date = '2015-06-18 09:00:00'

# Run inference
labeled_features = modeler.labeled_features
X_test = pd.get_dummies(labeled_features[labeled_features['datetime'] == inference_date].drop(['datetime', 'machineID', 'comp_to_fail'], axis=1))
y_pred = model.predict({'X': X_test})['predictions']
st.write(f"Predicted component to fail: {y_pred[0]}")
st.write(f"Actual component to fail: {labeled_features[labeled_features['datetime'] == inference_date]['comp_to_fail'].values[0]}")
