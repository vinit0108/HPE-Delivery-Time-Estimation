1. ModellingTrainingProcess.ipynb - A Jupyer Notebook outlining the whole modelling and testing process with explanation included. 

2. Testing.py/ Testing.ipynb - A python file to evaluate the performance of the model, the outputs are 1) results (A pandas DataFrame storing predicted and actual delivery time), 2) mean aboslute error of the delivery time (days), 3) percentage of delayed delivery which is the percentage of the predicted delivery time less than that of the actual. All the outputs are stored in a list and then stored in a pickle file. To execute the python file, the following command can be typed: 'execution_pipeline(df, training)' where 'df' is the pandas DataFrame of the testing dataset and 'training' is set to be True if the label exists in the testing dataset.

3. UpdateModel.py/ UpdateModel.ipynb - A python file to update the existing model with new datasets (either large or small). The feature selection is performed and the regression model is retrained. To execute the python file, the following command can be typed: 'retraining_pipeline(df, training)' where 'df' is the pandas DataFrame of the testing dataset and 'training' is set to be True if the label exists in the testing dataset.

4. ann.py - the modele storing the ANN regressor model

5. ANNRegressor_0.80quantile.h5py.index, ANNRegressor_0.80quantile.h5py.data-0000-of-0001 - the weights of the best trained model

6. deliverytime_scaler.pkl, quantity_scaler.pkl - pickle file storing standard scaler

7. selected_features.pkl - pickle file storing selected important features from gradient boosting

8. parameters.pkl - pickle file storing the parameters of the ANN regressor model

9. result.pkl - pickle file storing the prediction result, mean absolute error and percentage of delayed delivery

10. intermediate image - file storing all the intermediate result/ images of the modelling and training process


