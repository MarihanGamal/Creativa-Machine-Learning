import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------
# Loading the saved model and helper files
# ---------------------------------------------------
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')
scaled_flag = joblib.load('scaled_flag.pkl')
model_accuracy = joblib.load('model_accuracy.pkl')

st.set_page_config(page_title='Adult Income Prediction', layout='centered')

# ---------------------------------------------------
# Title
# ---------------------------------------------------
st.title('Adult Census Income Prediction App')
st.write('This app predicts whether a person earns **<=50K** or **>50K** per year, based on the Adult Census Income dataset.')

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
st.sidebar.header('About')
st.sidebar.write('This app uses a trained Machine Learning model to predict income category.')
st.sidebar.write(f'**Model Test Accuracy:** {model_accuracy*100:.2f}%')
st.sidebar.markdown('---')
st.sidebar.header('Input Options')
input_method = st.sidebar.radio('Choose input method:', ['Manual Input', 'Upload CSV'])


# helper function that takes raw inputs and builds the row of selected features
def build_feature_row(age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week,
                       workclass, education, marital_status, occupation, relationship, sex):
    # start with all zeros for the selected features
    row = dict.fromkeys(selected_features, 0)

    # fill in the numeric features directly
    row['age'] = age
    row['fnlwgt'] = fnlwgt
    row['education.num'] = education_num
    row['capital.gain'] = capital_gain
    row['capital.loss'] = capital_loss
    row['hours.per.week'] = hours_per_week

    # fill in the one-hot encoded categorical features (only if that column exists in selected_features)
    if 'workclass_Private' in row:
        row['workclass_Private'] = 1 if workclass == 'Private' else 0

    if 'education_Bachelors' in row:
        row['education_Bachelors'] = 1 if education == 'Bachelors' else 0

    if 'marital.status_Married-civ-spouse' in row:
        row['marital.status_Married-civ-spouse'] = 1 if marital_status == 'Married-civ-spouse' else 0
    if 'marital.status_Never-married' in row:
        row['marital.status_Never-married'] = 1 if marital_status == 'Never-married' else 0

    if 'occupation_Exec-managerial' in row:
        row['occupation_Exec-managerial'] = 1 if occupation == 'Exec-managerial' else 0
    if 'occupation_Prof-specialty' in row:
        row['occupation_Prof-specialty'] = 1 if occupation == 'Prof-specialty' else 0

    if 'relationship_Not-in-family' in row:
        row['relationship_Not-in-family'] = 1 if relationship == 'Not-in-family' else 0
    if 'relationship_Own-child' in row:
        row['relationship_Own-child'] = 1 if relationship == 'Own-child' else 0

    if 'sex_Male' in row:
        row['sex_Male'] = 1 if sex == 'Male' else 0

    # putting it into a dataframe with the exact same column order used during training
    X_row = pd.DataFrame([row])[selected_features]
    return X_row


def predict(X_row):
    if scaled_flag:
        X_row = pd.DataFrame(scaler.transform(X_row), columns=X_row.columns)
    pred = model.predict(X_row)[0]
    proba = model.predict_proba(X_row)[0]
    return pred, proba


# ---------------------------------------------------
# Manual Input Fields
# ---------------------------------------------------
if input_method == 'Manual Input':
    st.header('Enter Details')

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Age', min_value=17, max_value=90, value=35)
        fnlwgt = st.number_input('Fnlwgt (census weight)', min_value=10000, max_value=1500000, value=200000)
        education_num = st.slider('Education Number (years of education)', min_value=1, max_value=16, value=10)
        hours_per_week = st.slider('Hours per Week', min_value=1, max_value=99, value=40)
        capital_gain = st.number_input('Capital Gain', min_value=0, max_value=99999, value=0)
        capital_loss = st.number_input('Capital Loss', min_value=0, max_value=4356, value=0)

    with col2:
        workclass = st.selectbox('Workclass', ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
                                                 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
        education = st.selectbox('Education', ['Bachelors', 'Some-college', 'HS-grad', 'Masters', 'Doctorate',
                                                 '11th', '10th', '9th', '12th', '7th-8th', 'Assoc-acdm',
                                                 'Assoc-voc', 'Prof-school', '5th-6th', '1st-4th', 'Preschool'])
        marital_status = st.selectbox('Marital Status', ['Married-civ-spouse', 'Never-married', 'Divorced',
                                                           'Separated', 'Widowed', 'Married-spouse-absent',
                                                           'Married-AF-spouse'])
        occupation = st.selectbox('Occupation', ['Exec-managerial', 'Prof-specialty', 'Craft-repair',
                                                   'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct',
                                                   'Transport-moving', 'Handlers-cleaners', 'Farming-fishing',
                                                   'Tech-support', 'Protective-serv', 'Priv-house-serv',
                                                   'Armed-Forces'])
        relationship = st.selectbox('Relationship', ['Husband', 'Not-in-family', 'Own-child', 'Unmarried',
                                                       'Wife', 'Other-relative'])
        sex = st.selectbox('Sex', ['Male', 'Female'])

    if st.button('Predict'):
        X_row = build_feature_row(age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week,
                                   workclass, education, marital_status, occupation, relationship, sex)
        pred, proba = predict(X_row)

        st.subheader('Prediction Result')
        if pred == 1:
            st.success('This person is predicted to earn **>50K** per year.')
        else:
            st.info('This person is predicted to earn **<=50K** per year.')

        st.subheader('Prediction Probability')
        st.write(f'Probability of <=50K: {proba[0]*100:.2f}%')
        st.write(f'Probability of >50K : {proba[1]*100:.2f}%')
        st.progress(float(proba[1]))

# ---------------------------------------------------
# CSV Upload
# ---------------------------------------------------
else:
    st.header('Upload CSV File')
    st.write('Upload a CSV file with the same raw columns as the original dataset (age, workclass, fnlwgt, education, education.num, marital.status, occupation, relationship, race, sex, capital.gain, capital.loss, hours.per.week, native.country).')

    uploaded_file = st.file_uploader('Choose a CSV file', type=['csv'])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write('Preview of uploaded data:')
        st.dataframe(input_df.head())

        if st.button('Predict on Uploaded Data'):
            rows = []
            for _, r in input_df.iterrows():
                X_row = build_feature_row(
                    r.get('age', 0), r.get('fnlwgt', 0), r.get('education.num', 0),
                    r.get('capital.gain', 0), r.get('capital.loss', 0), r.get('hours.per.week', 0),
                    r.get('workclass', ''), r.get('education', ''), r.get('marital.status', ''),
                    r.get('occupation', ''), r.get('relationship', ''), r.get('sex', '')
                )
                rows.append(X_row)

            X_batch = pd.concat(rows, ignore_index=True)
            if scaled_flag:
                X_batch_final = pd.DataFrame(scaler.transform(X_batch), columns=X_batch.columns)
            else:
                X_batch_final = X_batch

            preds = model.predict(X_batch_final)
            probas = model.predict_proba(X_batch_final)[:, 1]

            result_df = input_df.copy()
            result_df['Predicted Income'] = np.where(preds == 1, '>50K', '<=50K')
            result_df['Probability of >50K'] = probas

            st.subheader('Prediction Results')
            st.dataframe(result_df)

# ---------------------------------------------------
# Model Accuracy (always shown at the bottom)
# ---------------------------------------------------
st.markdown('---')
st.write(f'**Model Test Accuracy:** {model_accuracy*100:.2f}%')
