import pandas as pd
from ml_model.ml_xgboost import XGBClassifierModel
from predictive_maintenance.modeler import PredictiveMaintananceModeler
if __name__ == '__main__':
    modeler = PredictiveMaintananceModeler()
    modeler.load_data() # in "real life" this would be loading from IoT sensor ETL pipeline
    model = XGBClassifierModel()
    model.load('xgb_classifier_1st_version.pkl')
    inference_date = '2015-06-18 09:00:00'

    # Run inference
    labeled_features = modeler.labeled_features
    X_test = pd.get_dummies(labeled_features[labeled_features['datetime'] == inference_date].drop(['datetime', 'machineID', 'comp_to_fail'], axis=1))
    y_pred = model.predict({'X': X_test})['predictions']
    print(f"Predicted component to fail: {y_pred[0]}")
    print(f"Actual component to fail: {labeled_features[labeled_features['datetime'] == inference_date]['comp_to_fail'].values[0]}")
