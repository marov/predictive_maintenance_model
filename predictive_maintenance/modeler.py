import pandas as pd
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from ml_model.ml_xgboost import XGBClassifierModel, XGBClassifierModeler
from .feature_transformers import (transformer_error_count,
                            transformer_maint_count,
                            transformer_telemetry_features_3h,
                            transformer_telemetry_features_24h,
                            transformer_labeled_features)

class PredictiveMaintananceModeler(XGBClassifierModeler):
    def __init__(self, train_test_split_ratio=0.3):
        super().__init__()
        self.model_class = XGBClassifierModel
        self.train_test_split_ratio = train_test_split_ratio
        self._error_maint_feat_pipeline = FeatureUnion([
            ('transformer_error_count', transformer_error_count),
            ('transformer_maint_count', transformer_maint_count),
        ])
        self._error_maint_feat_pipeline.set_output(transform='pandas')
        self._telemetry_feat_pipeline = FeatureUnion([
            ('transformer_telemetry_features_3h', transformer_telemetry_features_3h),
            ('transformer_telemetry_features_24h', transformer_telemetry_features_24h),
        ])
        self._telemetry_feat_pipeline.set_output(transform='pandas')
        self.labeled_features = None

    def load_data(self):
        """Generate labeled features and split data into train and test sets
        """
        if not self.labeled_features:
            # load data
            iot_pmfp_data_df = pd.read_feather('https://s3.us-west-1.amazonaws.com/aitomatic.us/pmfp-data/iot_pmfp_data.feather')
            iot_pmfp_labels_df = pd.read_feather('https://s3.us-west-1.amazonaws.com/aitomatic.us/pmfp-data/iot_pmfp_labels.feather')

            # add the labels to the error and maint record features
            error_maint_features = self._error_maint_feat_pipeline.fit_transform(iot_pmfp_labels_df).dropna()

            # add telemetry features
            telemetry_feat = self._telemetry_feat_pipeline.fit_transform(iot_pmfp_data_df).dropna()

            # merge telemetry and error/maint features into a single dataframe
            final_feat = telemetry_feat.merge(error_maint_features, on=['datetime', 'machineID'], how='left')

            # add labels to the data
            self.labeled_features = transformer_labeled_features.fit_transform((iot_pmfp_labels_df, final_feat)).dropna()

        X = pd.get_dummies(self.labeled_features.drop(['datetime', 'machineID', 'comp_to_fail'], axis=1))
        y = self.labeled_features['comp_to_fail']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.train_test_split_ratio, shuffle=False)

        prepared_data = {'X_train': X_train, 'y_train': y_train,
                        'X_test': X_test, 'y_test': y_test}
        return prepared_data