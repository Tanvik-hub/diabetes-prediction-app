# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open(r'C:/Users/Admin/OneDrive/Documents/model/trained_model.sav', 'rb'))

# sample input data for prediction
input_data = [4, 110, 92, 0, 0, 37.6, 0.191, 30]

# changing input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array for a single instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# make prediction
prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

# print prediction result
if prediction[0] == 0:
    print("The person is non-diabetic")
else:
    print("The person is diabetic")
