import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import pickle

model = pickle.load(open('model_V2.pkl', 'rb'))

with open('encoder_V2.pkl', 'rb') as pkl_file:
    encoder_dict = pickle.load(pkl_file)

def encode_features(df, encoder_dict):
    category_col = ['workclass', 'education', 'maritalstatus', 'occupation', 
                    'relationship', 'race', 'gender', 'nativecountry']
    for col in category_col:
        if col in encoder_dict:
            le = LabelEncoder()
            le.classes_ = np.array(encoder_dict[col], dtype=object)

            df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unkown')
            df[col] = le.transform(df[col])
    return df

def main():

    st.title("Income Predictor")
    age = st.slider("Age", 0, 100)
    workclass = st.selectbox("Working Class", ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov',
                                               'Local-gov', '?', 'Self-emp-inc', 'Without-pay',
                                               'Never-worked'])
    education = st.selectbox("Education", ['Bachelors', 'HS-grad', '11th', 'Masters', '9th',
                                           'Some-college', 'Assoc-acdm', 'Assoc-voc', '7th-8th',
                                           'Doctorate', 'Prof-school', '5th-6th', '10th', '1st-4th',
                                           'Preschool', '12th'])
    marital_status = st.selectbox("Marital Status", ['Never-married', 'Married-civ-spouse', 'Divorced',
                                                     'Married-spouse-absent', 'Separated', 'Married-AF-spouse',
                                                     'Widowed'])
    occupation = st.selectbox("Occupation", ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
                                             'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair',
                                             'Transport-moving', 'Farming-fishing', 'Machine-op-inspct',
                                             'Tech-support', '?', 'Protective-serv', 'Armed-Forces',
                                             'Priv-house-serv'])
    relationship = st.selectbox("Relationship", ['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried',
                                                 'Other-relative'])
    race = st.selectbox ("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
    gender = st.radio("Gender", ["Male", "Female"])
    native_country = st.selectbox("Native Country", ['United-States', 'Cuba', 'Jamaica', 'India', '?', 'Mexico',
                                                    'South', 'Puerto-Rico', 'Honduras', 'England', 'Canada',
                                                    'Germany', 'Iran', 'Philippines', 'Italy', 'Poland',
                                                    'Columbia', 'Cambodia', 'Thailand', 'Ecuador', 'Laos',
                                                    'Taiwan', 'Haiti', 'Portugal', 'Dominican-Republic',
                                                    'El-Salvador', 'France', 'Guatemala', 'China', 'Japan',
                                                    'Yugoslavia', 'Peru', 'Outlying-US(Guam-USVI-etc)', 'Scotland',
                                                    'Trinadad&Tobago', 'Greece', 'Nicaragua', 'Vietnam', 'Hong',
                                                    'Ireland', 'Hungary', 'Holand-Netherlands'])
    capital_gain = st.text_input("Capital Gain", "0")
    capital_loss = st.text_input("Capital Loss", "0")
    hours_per_week = st.text_input("Hours Per Week", "0")
    
    if st.button("Predict"):
        data = {'age': int(age), 'workclass': workclass, 'education': education, 'maritalstatus': marital_status, 
                'occupation': occupation, 'relationship': relationship, 'race': race, 'gender': gender, 
                'capitalgain': int(capital_gain), 'capitalloss': int(capital_loss), 'hoursperweek': int(hours_per_week),
                'nativecountry': native_country}
        df = pd.DataFrame([data])

        df = encode_features(df, encoder_dict)

        features_list = df.values
        prediction = model.predict(features_list)

        output = int(prediction[0])

        if output == 1:
            text = ">50K"
        else:
            text = "<=50K"
        
        st.success('Employee Income is {}'.format(text))

if __name__ == "__main__":
    main()