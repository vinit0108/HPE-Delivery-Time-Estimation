# **Project Description**
In this project, a product supplier company needs to create an algorithm that uses data science modelling to accurately predict the delivery time. In addition,the companies expect the model to avoid predicting the time to be less than the actual delivery time to prevent delayed delivery date and lead to customer dissatisfaction. Besides, the model should be able to update itself based on the new dataset. For more description of the problem, please refer to the attached pdf file ('Programming_Question.pdf').

# **Metholodgy Description**

### **Preprocess & Feature Engineering**
In the original data set, there are 13 attributes found, which are 'Order ID', 'Product Center', 'Product Sub Center', 'Order Product Quantity', 'Customer Expected Delivery Date', 'Delivery Priority Code', 'Product Shipping Point', 'Product Supplier', 'Customer Name', 'Order Type', 'Product ID', 'Product Type', 'Order Create Date' and 1 label used for regression - 'Delivery Time (days)'.

Before moving on to the model training, it is important to preprocess the data set and perform feature engineering to extract the utmost from the datset. The following few steps are executed:

Step 1 - The null item is first identified from the data set. From the dataset, it is noticed that there are samples without the label and the corresponding samples are not useful and removed from the dataset. Besides, there are also null item found in attribute 'Order ID' which is filled with '0' for ease of processig in the later steps.

Step 2 - Exploratory data analysis (EDA) is performed on the categorical attributes, which are 'Product Center', 'Product Sub Center', 'Delivery Priority Code', 'Product Shipping Point', 'Product Supplier', 'Order Type', and 'Product Type'. The unique elements of each categorical attribute is first identified and it is found that there is only 1 element in attribute 'Product Center' and thus it is relieved from EDA. From the EDA, the attributes 'Product Sub Center' and 'Order Type' can be removed since they are not important factor affecting the delivery time (days). EDA is an important step to remove redundant attributes with visualization which helps in prevent curse of dimensionality.

![ALT text](https://raw.githubusercontent.com/ChongAih/HPE-Delivery-Time-Estimation/blob/image/EDA.png)

Step 3 - From the dataset, there are two datetime attributes available - 'Customer Expected Delivery Date' and 'Order Create Date'. The attribute 'Customer Expected Delivery Date' is deemed not useful since the expectation does not affect the actual delivery time. In the other hand, the attribute 'Order Create Date' is important since the schedule of delivery only starts after the order is created. To extract information from the attribute, the feature engineering is performed and the new attributes 'Month', 'Day' and 'Weekend' are created as tt is speculated that there might be peak period during certain month, day or on weekend.

Step 4 - The unwanted attributes are now removed from the dataset to prevent curse of dimensionality.

Step 5 - The categorical attributes are one-hot encoded and the first column of each categorical attribute is removed to prevent possible problem arises from the multicollinearity.


### **Feature Selection**
With the preprocessed data set, feature selection is executed to identify the truly important feature. In this project, the ensembled tree method - gradient boosting is used to study the importance of each feature. Huber loss is selected in the gradient boosting as it is found from EDA that there are numbers of possible outlier; as a combination of mean square error and mean aboslute error, huber loss can outperform in such case. Also note that, the scaling is not performed since the performance of tree based machine learning method is independent of the scaling. The top 25 features are selected.

![alt text](https://github.com/ChongAih/HPE-Delivery-Time-Estimation/blob/image/importance.png?raw=true)

It is noticed that the attributes 1) 'Product Shipping Point', 2) 'Product Type', 3) 'Order Product Quantity', 4) 'Delivery Priority Code' make up up to 80% of the importance. It can be explained rationally - 1) the location of shipping point matters since it will take longer time to travel out from a remote area; 2) different type of product might take different time to produce; 3) the larger quantity order, the more time needed to produce; 4) the higher the priority the faster the delivery

### **Modeling**
In the proposed approach, the problem is treated as a multi-objective problems - 1) the accurate estimation of delivery time; 2) the prediction of delivery time being less than actual delivery time must be avoided to prevent delayed delivery and lead to customer dissatisfaction. In view of the objective functions, a 3 layer artificial neural network based regression model is selected due to the ease of modifying the loss function for training. In order to minimize the customer dissatisfaction, a quantile loss as depicted in image below is deployed to ensure that most of the prediction of delivery time is always higher than the actual delivery time and therefore lower the delayed delivery and the customer dissatification. For instance, 95% quantile implies that the model will try its best effort to ensure 95% of the time, the prediction value is always higher than the actual value.

![alt text](https://github.com/ChongAih/HPE-Delivery-Time-Estimation/blob/image/quantile_loss.png?raw=true)

Nonetheless, it is a multi-objective problem, an appropriate amount of attention shall be paid to the prediction accuracy. Thus, the performance of the model with different values of quantile are evaluated with 2 proposed metrics - 1) the mean absolute error of the delivery time; 2) the percentage of delayed delivery. From the recorded result, a pareto front is plotted and it is found that quantile value ranging from 75% to 85% returns the best balanced resut. In multi-objective problem, the pareto optimality can only be achieved where no individual or preference criterion can be better off without making at least one individual or preference criterion worse off. In the project, mean absolute error and percentage of delayed delivery are traded off across different quantile values.

![alt text](https://github.com/ChongAih/HPE-Delivery-Time-Estimation/blob/image/pareto.png?raw=true)

**Please refer to the 'ModellingTrainingProcess.ipynb' for the detailed explanation of the modelling and training process.**

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


