import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer

def telemetry_features_3h(iot_pmfp_data_df):
    # Calculate mean values for telemetry features
    temp = []
    fields = ['volt', 'rotate', 'pressure', 'vibration']
    for col in fields:
        temp.append(pd.pivot_table(iot_pmfp_data_df,
                                index='datetime',
                                columns='machineID',
                                values=col).resample('3H', closed='left', label='right').mean().unstack())
    telemetry_mean_3h = pd.concat(temp, axis=1)
    telemetry_mean_3h.columns = [i + 'mean_3h' for i in fields]
    telemetry_mean_3h.reset_index(inplace=True)

    # repeat for standard deviation
    temp = []
    for col in fields:
        temp.append(pd.pivot_table(iot_pmfp_data_df,
                                index='datetime',
                                columns='machineID',
                                values=col).resample('3H', closed='left', label='right').std().unstack())
    telemetry_sd_3h = pd.concat(temp, axis=1)
    telemetry_sd_3h.columns = [i + 'sd_3h' for i in fields]
    telemetry_sd_3h.reset_index(inplace=True)
    telemetry_3h = pd.concat([telemetry_mean_3h, telemetry_sd_3h.iloc[:, 2:6]], axis=1)
    telemetry_3h = telemetry_3h.merge(iot_pmfp_data_df[['datetime', 'machineID', 'model', 'age']], on=['datetime', 'machineID'], how='left')
    return telemetry_3h
transformer_telemetry_features_3h = FunctionTransformer(telemetry_features_3h, validate=False)

def telemetry_features_24h(iot_pmfp_data_df):
    temp = []
    fields = ['volt', 'rotate', 'pressure', 'vibration']
    for col in fields:
        temp.append(pd.pivot_table(iot_pmfp_data_df,index='datetime',
                                                columns='machineID',
                                                values=col).rolling(24).mean().resample('3H',
                                                                                    closed='left',
                                                                                    label='right').first().unstack())
    telemetry_mean_24h = pd.concat(temp, axis=1)
    telemetry_mean_24h.columns = [i + 'mean_24h' for i in fields]
    telemetry_mean_24h.reset_index(inplace=True)
    telemetry_mean_24h = telemetry_mean_24h.loc[-telemetry_mean_24h['voltmean_24h'].isnull()]

    # repeat for standard deviation
    temp = []
    fields = ['volt', 'rotate', 'pressure', 'vibration']
    for col in fields:
        temp.append(pd.pivot_table(iot_pmfp_data_df,index='datetime',
                                                columns='machineID',
                                                values=col).rolling(24).std().resample('3H',
                                                                                    closed='left',
                                                                                    label='right').first().unstack())
    telemetry_sd_24h = pd.concat(temp, axis=1)
    telemetry_sd_24h.columns = [i + 'sd_24h' for i in fields]
    telemetry_sd_24h = telemetry_sd_24h.loc[-telemetry_sd_24h['voltsd_24h'].isnull()]
    telemetry_sd_24h.reset_index(inplace=True)
    return pd.concat([telemetry_mean_24h.iloc[:, 2:6], telemetry_sd_24h.iloc[:, 2:6]], axis=1)
transformer_telemetry_features_24h = FunctionTransformer(telemetry_features_24h, validate=False)

def error_count(iot_pmfp_labels_df):
    temp = []
    fields = ['error%d' % i for i in range(1,6)]
    for col in fields:
        temp.append(pd.pivot_table(iot_pmfp_labels_df,
                                                index='datetime',
                                                columns='machineID',
                                                values=col).rolling(24).sum().resample('3H',
                                                                                closed='left',
                                                                                label='right').first().unstack())
    error_count = pd.concat(temp, axis=1)
    error_count.columns = [i + 'count' for i in fields]
    error_count.reset_index(inplace=True)
    # error_count = error_count.dropna()
    return error_count
transformer_error_count = FunctionTransformer(error_count, validate=False)

def maint_count(iot_pmfp_labels_df):
    temp = []
    fields = ['maint_comp%d' % i for i in range(1,5)]
    for col in fields:
        temp.append(pd.pivot_table(iot_pmfp_labels_df,
                                                index='datetime',
                                                columns='machineID',
                                                values=col).expanding().sum().resample('3H',
                                                                                closed='left',
                                                                                label='right').first().unstack())
    maint_count = pd.concat(temp, axis=1)
    maint_count.columns = [i + '_count' for i in fields]
    maint_count.reset_index(inplace=True)
    # maint_count = maint_count.dropna()
    return maint_count.drop(['datetime', 'machineID'], axis=1)
transformer_maint_count = FunctionTransformer(maint_count, validate=False)

def labeled_features_add(data: tuple):
    """Add labels to the data:
    - Add a comp_to_fail column from the failure_comp columns for individual component failures
    - Merge the error and maintenance features
    - Add a none column for no failure
    - When a component marked as failed, backfill all the rows for the last 24 hours. Assumption is that the component has 24 hours to be replaced.
    - Remaining rows with no failure are marked as 'none'
    """
    iot_pmfp_labels_df, final_feat = data

    iot_pmfp_labels_df['none'] = np.where(iot_pmfp_labels_df['failure'] == False, 1, 0)  # 0 is failure, 1 is no failure
    iot_pmfp_labels_df['comp_to_fail'] = iot_pmfp_labels_df[['failure_comp1','failure_comp2','failure_comp3','failure_comp4', 'none']].idxmax(axis=1)
    iot_pmfp_labels_df['comp_to_fail'] = np.where(iot_pmfp_labels_df['comp_to_fail'] == 'none', np.nan, iot_pmfp_labels_df['comp_to_fail'])
    iot_pmfp_labels_df['comp_to_fail'] = iot_pmfp_labels_df['comp_to_fail'].astype('category')

    labeled_features = final_feat.merge(iot_pmfp_labels_df[['datetime', 'machineID', 'comp_to_fail']],
                                                   on=['datetime', 'machineID'], how='left')

    labeled_features = labeled_features.fillna(method='bfill', limit=7) # fill backward up to 24h
    labeled_features['comp_to_fail'] = labeled_features['comp_to_fail'].cat.add_categories(
        'none')
    labeled_features['comp_to_fail'] = labeled_features['comp_to_fail'].fillna("none")
    return labeled_features
transformer_labeled_features = FunctionTransformer(labeled_features_add, validate=False)
