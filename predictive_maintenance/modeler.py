import pandas as pd
from sklearn.pipeline import FeatureUnion
# from sklearn.model_selection import train_test_split, TimeSeriesSplit
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
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.split_date = None

    def load_data(self):
        """Generate labeled features and split data into train and test sets
        """
        if not self.X_train or not self.X_test or not self.y_train or not self.y_test:
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

        X = pd.get_dummies(self.labeled_features.drop(['machineID', 'comp_to_fail'], axis=1)) # we need 'datetime', for splitting data
        y = self.labeled_features['comp_to_fail']

        # split data into train and test sets by 'datetime' column at self.train_test_split_ratio
        dates = X['datetime'].unique()
        split_index = int(len(dates) * self.train_test_split_ratio)
        self.split_date = dates[split_index]
        self.X_train, self.X_test = X[X['datetime'] < self.split_date], X[X['datetime'] >= self.split_date]
        self.y_train, self.y_test = y[X['datetime'] < self.split_date], y[X['datetime'] >= self.split_date]
        self.X_train, self.X_test = self.X_train.drop(['datetime'], axis=1), self.X_test.drop(['datetime'], axis=1)

        prepared_data = {'X_train': self.X_train, 'y_train': self.y_train,
                        'X_test': self.X_test, 'y_test': self.y_test}
        return prepared_data