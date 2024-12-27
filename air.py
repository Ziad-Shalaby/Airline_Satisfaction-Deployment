import streamlit as st 
import pandas as pd
import numpy as np
import pickle
from collections import Counter

st.set_page_config(layout='wide')
st.title('🛫Airline Passenger Satisfaction')

Knn = pickle.load(open('knn.pkl', 'rb'))
Svm = pickle.load(open('svm.pkl', 'rb'))
Naive = pickle.load(open('Naive.pkl', 'rb'))
Decision = pickle.load(open('decision_tree.pkl', 'rb'))
Gradient = pickle.load(open('GradientBoostingClassifier.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
pt = pickle.load(open('pt.pkl', 'rb'))

input_names = [
    'Gender', 'Age', 'Flight Distance', 'Inflight wifi service',
    'Departure/Arrival time convenient', 'Gate location', 'Food and drink',
    'Online boarding', 'Seat comfort', 'Inflight entertainment',
    'On-board service', 'Leg room service', 'Baggage handling',
    'Checkin service', 'Inflight service', 'Cleanliness',
    'Departure Delay in Minutes'
]

cat_features = ['Gender']

def user_input():
    features = {}

    col1, col2 = st.columns(2)

    with col1:
        features['Age'] = st.number_input('👶Age', min_value=1, max_value=100, value=10)
        features['Flight Distance'] = st.number_input('🛣️Flight Distance')
        features['Inflight wifi service'] = st.number_input('📶Inflight wifi service', min_value=0, max_value=5, value=2)
        features['Departure/Arrival time convenient'] = st.number_input('⌚Departure/Arrival time convenient', min_value=0, max_value=5, value=2)
        features['Gate location'] = st.number_input('🛅Gate location', min_value=0, max_value=5, value=2)
        features['Food and drink'] = st.number_input('🍔Food and drink', min_value=0, max_value=5, value=2)
        features['Departure Delay in Minutes'] = st.number_input('⌚Departure Delay in Minutes', min_value=0, max_value=5, value=2)
        features['Inflight service'] = st.number_input('🤗Inflight service', min_value=0, max_value=5, value=2)
        features['Gender'] = st.selectbox('🧑Gender', options=['Male', 'Female'])
    with col2:
        features['Online boarding'] = st.number_input('📶Online boarding', min_value=0, max_value=5, value=2)
        features['Seat comfort'] = st.number_input('💺Seat comfort', min_value=0, max_value=5, value=2)
        features['On-board service'] = st.number_input('🌀On-board service', min_value=0, max_value=5, value=2)
        features['Leg room service'] = st.number_input('🌀Leg room service', min_value=0, max_value=5, value=2)
        features['Baggage handling'] = st.number_input('🛄Baggage handling', min_value=0, max_value=5, value=2)
        features['Checkin service'] = st.number_input('✅Checkin service', min_value=0, max_value=5, value=2)
        features['Cleanliness'] = st.number_input('🫧Cleanliness', min_value=0, max_value=5, value=2)
        features['Inflight entertainment'] = st.number_input('🎮Inflight entertainment', min_value=0, max_value=5, value=2)

    return features

user_features = user_input()

features_list = []
for col in input_names:
    value = user_features[col]

    if col in cat_features:
        le = pickle.load(open(f'PKl Files/encoder_{col}.pkl', 'rb'))
        transformed_value = le.transform(np.array([[value]]))
        features_list.append(transformed_value.item())
    else:
        features_list.append(value)

features_array = np.array(features_list).reshape(1, -1)
feature_trans = pt.transform(features_array)
features_scaled = scaler.transform(feature_trans)

if 'y_pred' not in st.session_state:
    st.session_state.y_pred = []

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    if st.button('Knn'):
        y_pred_model_log = Knn.predict(features_scaled)[0]
        st.session_state.y_pred.append(y_pred_model_log)
        if y_pred_model_log == 1:
            st.success('Knn: satisfied')
        else:
            st.error('Knn: neutral or dissatisfied')

with col2:
    if st.button('SVM'):
        y_pred_model_SVM = Svm.predict(features_scaled)[0]
        st.session_state.y_pred.append(y_pred_model_SVM)
        if y_pred_model_SVM == 1:
            st.success('SVM: satisfied')
        else:
            st.error('SVM: neutral or dissatisfied')

with col3:
    if st.button('GradientBoosting'):
        y_pred_model_GradientBoosting = Gradient.predict(features_scaled)[0]
        st.session_state.y_pred.append(y_pred_model_GradientBoosting)
        if y_pred_model_GradientBoosting == 1:
            st.success('Gradient: satisfied')
        else:
            st.error('Gradient: neutral or dissatisfied')

with col4:
    if st.button('Naive Bayes'):
        y_pred_model_Naive = Naive.predict(features_scaled)[0]
        st.session_state.y_pred.append(y_pred_model_Naive)
        if y_pred_model_Naive == 1:
            st.success('Naive Bayes: satisfied')
        else:
            st.error('Naive Bayes: neutral or dissatisfied')

with col5:
    if st.button('Decision Tree'):
        y_pred_model_Decision = Decision.predict(features_scaled)[0]
        st.session_state.y_pred.append(y_pred_model_Decision)
        if y_pred_model_Decision == 1:
            st.success('Decision Tree: satisfied')
        else:
            st.error('Decision Tree: neutral or dissatisfied')

with col6:
    if st.button('Final Voting From Models'):
        if len(st.session_state.y_pred) == 5:
            y_pred_final = Counter(st.session_state.y_pred)
            if y_pred_final.most_common(1)[0][0] == 1:
                st.success('Final Prediction: satisfied')
            else:
                st.error('Final Prediction: neutral or dissatisfied')
            del st.session_state.y_pred
        else:
            st.error('Press on each model button first')
