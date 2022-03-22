import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

header = st.container()
dataset = st.container()
feature = st.container()
training = st.container()

with header:
    st.title('Welcome to the Heart Attack predictor')
    st.text('This AI project predicts the possibility of Heart Attack in next Ten Years')
    st.markdown('This web is developed by  **Suhas C B**')

with dataset:
    st.title('This is the dataset sample of few patients')

    df = pd.read_csv('framingham.csv')
    df_new = df[
        ['age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'totChol', 'sysBP', 'diaBP', 'heartRate', 'TenYearCHD']]
    st.write(df_new.head())
    df_new['cigsPerDay'] = df_new['cigsPerDay'].fillna(df['cigsPerDay'].median())
    df_new['BPMeds'] = df_new['BPMeds'].fillna(0)
    df_new['totChol'] = df_new['totChol'].fillna(df['totChol'].mean())
    df_new['heartRate'] = df_new['heartRate'].fillna(df['heartRate'].mean())

    df_new['currentSmoker'] = df_new['currentSmoker'].astype('category')
    df_new['BPMeds'] = df_new['BPMeds'].astype('category')
    df_new['TenYearCHD'] = df_new['TenYearCHD'].astype('category')

    Y = df_new[['TenYearCHD']]
    X = df_new[['age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'totChol', 'sysBP', 'diaBP', 'heartRate']]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)

    X_train_array = np.array(X_train)
    Y_train_array = np.array(Y_train)

    lr = LogisticRegression()

    model1 = lr.fit(X_train, Y_train)

    sm = SMOTE(random_state=1234)

    X_train_new, Y_train_new = sm.fit_resample(X_train_array, Y_train_array)

    lr_new = LogisticRegression()

    model2 = lr_new.fit(X_train_new, Y_train_new)

    Y_test['Pred_values_new'] = model2.predict(X_test)

    accu = accuracy_score(Y_test['TenYearCHD'], Y_test['Pred_values_new'])
    accu1 = accu * 100
    accuracy = "{:.2f}".format(accu1)

    heartRate = pd.DataFrame(df_new[['cigsPerDay', 'heartRate', 'sysBP']]).head(300)

    st.markdown('**Scatter plot of SMOKERS Heart Rate, Cigs per Day and BP**')

    chart = alt.Chart(heartRate).mark_circle().encode(
        x='cigsPerDay', y='heartRate', size='sysBP', color='sysBP'
    )
    st.altair_chart(chart, use_container_width=True)

with feature:
    st.title('Feature of the predictor')
    st.markdown('* **This model was built using Logistic Regression**')
    st.markdown("* **This model has {0} % accuracy.**".format(accuracy))

with training:
    st.title('Time to predict')
    st.text('Fill the parameters to predict the possibility of Heart Attack in next Ten Years')

    sel_col, disp_col = st.columns(2)

    age = sel_col.slider('What is your age?', min_value=18, max_value=80, value=20)
    current_smoker = sel_col.selectbox('Are you an active smoker?', options=['Yes', 'No'])
    if current_smoker == 'Yes':
        current_smoker = 1
        cigs_day = sel_col.slider('How many Cigarettes do you smoke per day?', min_value=0, max_value=70, value=0)
    else:
        current_smoker = 0
    BP_meds = sel_col.selectbox('Do you take any BP related medications?', options=['Yes', 'No'])
    cholesterol = sel_col.slider('What is your average cholesterol?', min_value=100, max_value=700, value=300, step=25)
    sys_BP = sel_col.slider('What is your Systolic Blood Pressure?', min_value=80, max_value=300, value=100, step=2)
    dia_BP = sel_col.slider('What is your Diastolic Blood Pressure?', min_value=40, max_value=150, value=100)
    heart_rate = sel_col.slider('What is your heart rate?', min_value=40, max_value=150, value=100)

    if BP_meds == 'Yes':
        BP_meds = 1
    else:
        BP_meds = 0

    feature = [[age, current_smoker, cigs_day, BP_meds, cholesterol, sys_BP, dia_BP, heart_rate]]

    if st.button('Submit'):
        prediction = model2.predict(feature)
        if prediction == 0:
            answer = "There is no chance of Heart Attack for you within the next 10 years"
        else:
            answer = "There is a risk of getting Heart Attack for you within the next 10 years"
        sel_col.subheader('Your health result is:')
        sel_col.write(answer)