#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ann import ANNRegressor_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as MAE

sns.set(style='whitegrid')


# # Testing process
# 
# 1. **Step 1** - In the modelling and training process, the important features/ attributes have been selected and saved and so as the ANN regressor model parameters and the scalers. Load those saved parameters to reconstruct the modelling environment.
# 
# 2. **Step 2** - Load the test dataset and perform preprocessing and feature engineering based on the saved selected features.
# 
# 3. **Step 3** - Standardize the dataset based on the saved scalers
# 
# 4. **Step 4** - Reconstruct the ANN regressor model and predict the result based on the preprocessed dataset.
# 
# 5. **Step 5** - Save the prediction results and evaluation metrics in list and store in a pickle file with details as follows:
#     * element 1 - result (pandas DataFrame) - prediction and actual delivery time
#     * element 2 - mae (numpy float) - mean absolute error of the delivery time
#     * element 3 - percentage_delayed_delivery (numpy float) - the percentage of prediction < actual delivery time
#     
# 6. Note that the mae and percentage of delayed delivery is quite high and the model can be further fine tuned with larger dataset to make it less overfit/ generalized.

# **Load respective saved pickle files**

# In[2]:


# Load the model parameters
with open('parameters.pkl', 'rb') as file:
    units, num_features, file_dir, epochs, batch_size, quantile = pickle.load(file)
    
# Load the important features/ attributes extracted
with open('selected_features.pkl', 'rb') as file:
    selected_features = pickle.load(file)
    
# Load the scaler for 'Order Product Quantity' &'Delivery Time (days)'
with open('quantity_scaler.pkl', 'rb') as file:
    quantity_scaler = pickle.load(file)
    
with open('deliverytime_scaler.pkl', 'rb') as file:
    deliverytime_scaler = pickle.load(file)


# **Function - Preprocess & feature engineering**

# In[3]:


# Set the parameter 'training' to True if the label exists

def preprocess_featureengineering(df, training):
    
    # Identify and handle null items
    if training: # The label is only available in training
        idx_null = np.where(df['Delivery Time (days)'].isnull()==1)[0]
        df.drop(idx_null, axis=0, inplace=True) # Remove the samples without the labels
        df.reset_index(drop=True, inplace=True) # Reset index
    df.fillna('0', inplace=True) # Fill null item with '0' for attribute 'Order Type'
    
    # Feature engineering with attribute 'Order Create Date'
    df['Month'] = df['Order Create Date'].apply(lambda x: x.month)
    df['Day'] = df['Order Create Date'].apply(lambda x: x.day)
    df['Weekend'] = df['Order Create Date'].apply(lambda x: 1 if x.weekday() >=5 else 0)    
    
    # Remove the unwanted columns
    columns_to_remove = ['Customer Expected Delivery Date', 'Order ID', 'Product Center', 
                         'Product Sub Center', 'Customer Name', 'Order Type', 
                         'Order Create Date', 'Product ID']
    df.drop(columns_to_remove, axis=1, inplace=True)
    
    # One hot encode categorical attributes
    columns_categorical = ['Delivery Priority Code', 'Product Shipping Point', 
                           'Product Supplier', 'Product Type', 'Month', 
                           'Day', 'Weekend']
    df = pd.get_dummies(df, columns=columns_categorical, drop_first=True)
    
    # Create important attributes/ selected features if it is missing from the current dataset
    # Assign value to be zeros
    for selected_feature in selected_features:
        if selected_feature not in df.columns.tolist():
            df[selected_feature] = 0   
    
    return df


# **Function - Standardization of numerical features**

# In[4]:


# Set the parameter 'training' to True if the label exists

def standardize(df, training):
    
    # Standardization of delivery time and order product quantity
    x_categorical = df[selected_features].drop('Order Product Quantity', axis=1, inplace=False).values
    x_numerical = df['Order Product Quantity'].values
    x_numerical = quantity_scaler.transform(np.expand_dims(x_numerical, axis=1))
    x_test = np.concatenate((x_categorical, x_numerical), axis=1)
    
    if training:
        y = df['Delivery Time (days)']
        y_test = deliverytime_scaler.transform(np.expand_dims(y, axis=1))
        # return scaled features/ attributes and labels
        return x_test, y_test
    
    return x_test


# **Function - Execution pipeline**

# In[5]:


# Set the parameter 'training' to True if the label exists

def execution_pipeline(df, training):
    
    df = preprocess_featureengineering(df, training)
    
    # Predict and save results and evaluation metrics
    x_test, y_test = standardize(df, training) # Scaled features & labels
    y_test_pred = model.predict(x_test) # rounded unscaled prediction time in days
    
    # Note that the prediction return is unscaled while the actual needs to be scaled back
    result = pd.DataFrame({'Predicted': np.squeeze(y_test_pred), 
                           'Actual': np.squeeze(deliverytime_scaler.inverse_transform(y_test))})
    print(result.head(), '\n')
    mae, percentage_delayed_delivery = model.evaluate(x_test, y_test, plot=False)
    
    # The saved pickle files store three elements
    # 1) result (pandas DataFrame) - prediction and actual delivery time
    # 2) mae (numpy float) - mean absolute error of the delivery time
    # 3) percentage_delayed_delivery (numpy float) - the percentage of prediction < actual delivery time
    with open('result.pkl', 'wb') as file:
        pickle.dump([result, mae, percentage_delayed_delivery], file)    


# **Initialization of ANN regressor model**

# In[6]:


model = ANNRegressor_model(units, num_features, quantile, 
                           file_dir, epochs, batch_size,
                           deliverytime_scaler)


# **Main**

# In[7]:


if __name__ == '__main__':
    
    # Read file and preprocess & feature engineer
    df = pd.read_excel('data_test.xlsx')
    
    # Set the parameter 'training' to True if the label exists
    training = True
    
    # Execution pipeline
    execution_pipeline(df, training)

