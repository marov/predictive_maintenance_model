import pandas as pd
from ml_model.ml_xgboost import XGBClassifierModel
from predictive_maintenance.modeler import PredictiveMaintananceModeler
if __name__ == '__main__':
    modeler = PredictiveMaintananceModeler(train_test_split_ratio=0.5)
    modeler.load_data() # in "real life" this would be loading from IoT sensor ETL pipeline
    model = XGBClassifierModel()
    model.load('xgb_classifier_1st_version.pkl')
    inference_date = '2015-09-01 09:00:00'

    # Run inference
    # labeled_features = modeler.labeled_features

    X = modeler.labeled_features[modeler.labeled_features['datetime'] == inference_date]
    # X_test = modeler.X_test[modeler.X_test['datetime'] == inference_date].drop(['datetime'], axis=1)
    # y_pred = model.predict({'X':X_test})['predictions']
    y_pred = model.predict({'X':
                             pd.get_dummies(X.drop(['datetime', 'machineID', 'comp_to_fail'], axis=1))})['predictions']
    X['comp_to_fail_pred'] = y_pred
    print(f"Predicted component to fail:\n {X[['machineID', 'comp_to_fail_pred']]}")
    print(f"Actual component to fail:\n {X[['machineID', 'comp_to_fail']]}")

