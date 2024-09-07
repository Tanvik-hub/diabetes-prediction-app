# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 21:57:43 2024

@author: Admin
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open(r'C:/Users/Admin/OneDrive/Documents/model/trained_model.sav', 'rb'))


def diabetes_prediction(input_data):
   
    # changing input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array for a single instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # make prediction
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    # print prediction result
    if prediction[0] == 0:
        return "The person is non-diabetic"
    else:
        return "The person is diabetic"
    
def main():
    st.title('Diabetes Prediction Web App')
    
    
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure')
    SkinThickness = st.text_input('Skin Thickness') 
    Insulin = st.text_input('Insulin Level') 
    BMI = st.text_input('BMI') 
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function') 
    Age = st.text_input('Age')
    
    diagnosis=''
    if st.button('Diabetes Test Result'):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    
    st.success(diagnosis)
    
    
    
if __name__=='__main__':
    main()
      
    
    
    
    
    
    
    
    
    
    