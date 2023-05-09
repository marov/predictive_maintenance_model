import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from ml_model.ml_xgboost import XGBClassifierModel
from predictive_maintenance.modeler import PredictiveMaintananceModeler

"""
# Predictive Maintenance :wrench: :factory:
"""

@st.cache_resource
def load_data():
    modeler = PredictiveMaintananceModeler()
    modeler.load_data() # in "real life" this would be loading from IoT sensor ETL pipeline
    return modeler.labeled_features, modeler.split_date

# Load data and model
labeled_features, split_date = load_data()
model = XGBClassifierModel()
model.load('xgb_classifier_1st_version.pkl')

# inference_date = '2015-09-01 09:00:00' was chosen as default, simly because it had a failure in a component

# Widget to select year, month, day, hour, minute
date = st.sidebar.date_input('Select date', pd.to_datetime('2015-09-01'),
                                min_value=split_date,
                                max_value=labeled_features['datetime'].max()
                            )
time = st.sidebar.time_input('Select time', pd.to_datetime('09:00:00'), step=pd.to_timedelta('3 hour'))
inference_date = pd.to_datetime(str(date) + ' ' + str(time))
st.markdown(f"<h3 style='color: blue;'>{inference_date}</h3>", unsafe_allow_html=True)

# Run inference
X_test = pd.get_dummies(labeled_features[labeled_features['datetime'] == inference_date]
                        .drop(['datetime', 'machineID', 'comp_to_fail'], axis=1))
y_pred = model.predict({'X': X_test})['predictions']

# Dropdown to select machineID
machineID = st.selectbox('Select machineID', labeled_features['machineID'].unique())

# Display alert of y_pred[0]
if y_pred[machineID-1] == 'none':
    st.success('No component is predicted to fail')
else:
    st.error(f'Component **{y_pred[machineID-1][-1]}** of machine {machineID} is predicted to fail')

# Plot failures in the last 5 days and the next 5 days
st.subheader(f"Failures in the last 5 days and the next 5 days :chart_with_upwards_trend:")
points = []
for i in range(-5, 6):
    inference_date = pd.to_datetime(str(date) + ' ' + str(time)) + pd.to_timedelta(str(i) + ' day')
    X_test = pd.get_dummies(labeled_features[labeled_features['datetime'] == inference_date]
                            .drop(['datetime', 'machineID', 'comp_to_fail'], axis=1))
    y_pred = model.predict({'X': X_test})['predictions']
    points.append([inference_date, y_pred[machineID-1]])
points = pd.DataFrame(points, columns=['datetime', 'comp_to_fail'])
# display the chart with the predictions, mark the current date with a vertical line
fig, ax = plt.subplots()
ax.plot(points['datetime'], points["comp_to_fail"], label='Timeline of failures')
ax.axvline(pd.to_datetime(str(date) + ' ' + str(time)), color='r', linestyle='--', label='Current date')
ax.legend()
st.pyplot(fig)
