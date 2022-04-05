import streamlit as st
import pickle
import pandas as pd
import numpy as np

def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('train_cols.pkl', 'rb') as file:
        train_cols = pickle.load(file)
    return model, train_cols

regressor, train_cols = load_model()

def show_predict_page():
    st.title("Crystal Ball-park")
    st.write("""### Predicting salaries for Tech jobs worldwide!""")
    st.write("""#### Let us first get some information about the job!""")

    countries = (
        "United States of America",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    educations = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    age = ('25-34 years old', '35-44 years old', '45-54 years old',
       '18-24 years old', '55-64 years old', '65 years or older',
       'Under 18 years old', 'Prefer not to say')

    orgsizes = ('10 to 19 employees', '1,000 to 4,999 employees',
       '100 to 499 employees', '500 to 999 employees',
       '5,000 to 9,999 employees', '2 to 9 employees',
       '20 to 99 employees', '10,000 or more employees', 'I don’t know',
       'Just me - I am a freelancer, sole proprietor, etc.')

    genders = ('Man', 'Woman', 'Prefer not to say', 'Non-binary')

    country = st.selectbox("Country of job", countries)
    education = st.selectbox("Level of Education", educations)
    age = st.selectbox("Age of candidate", age)
    orgsize = st.selectbox("Company Size", orgsizes)
    gender = st.selectbox("Gender", genders)

    experience = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        X_df = pd.DataFrame({'Country':country,'EdLevel':education,'YearsCodePro':experience,'Age':age,
             'Gender':gender,'OrgSize':orgsize},index=[0])
        
        X_ohe = pd.get_dummies(X_df)
        X_ohe = X_ohe.reindex(columns = train_cols, fill_value=0)
        salary = regressor.predict(X_ohe)
        st.subheader(f"The estimated salary is ${salary[0]:,.2f}")
