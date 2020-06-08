# **Project Description**
In this project, a product supplier company needs to create an algorithm that uses data science modelling to accurately predict the delivery time. In addition,the companies expect the model to avoid predicting the time to be less than the actual delivery time to prevent delayed delivery date and lead to customer dissatisfaction. Besides, the model should be able to update itself based on the new dataset. For more description of the problem, please refer to the attached pdf file ('Programming_Question.pdf').

# **Modelling Description**
In the proposed approach, the problem is treated as a multi-objective problems - 1) the accurate estimation of delivery time; 2) the prediction of delivery time being less than actual delivery time must be avoided to prevent delayed delivery and lead to customer dissatisfaction. In view of the objective functions, a 3 layer artificial neural network based regression model is selected due to the ease of modifying the loss function for training. In order to minimize the customer dissatisfaction, a quantile loss as depicted in image below is deployed to ensure that most of the prediction of delivery time is always higher than the actual delivery time and therefore lower the delayed delivery and the customer dissatification. For instance, 95% quantile implies that the model will try its best effort to ensure 95% of the time, the prediction value is always higher than the actual value. 

Nonetheless, it is a multi-objective problem, an appropriate amount of attention shall be paid to the prediction accuracy. Thus, the performance of the model with different values of quantile are evaluated with 2 proposed metrics - 1) the mean absolute error of the delivery time; 2) the percentage of delayed delivery. From the recorded result, a pareto front is plotted and it is found that quantile value ranging from 75% to 85% returns the best balanced resut. In multi-objective problem, the pareto optimality can only be achieved where no individual or preference criterion can be better off without making at least one individual or preference criterion worse off. In the project, mean absolute error and percentage of delayed delivery are traded off across different quantile values.

Please refer to the 'ModellingTrainingProcess.ipynb' for the detailed explanation of the modelling and training process. 

# **File description**
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


