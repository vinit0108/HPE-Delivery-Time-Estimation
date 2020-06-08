#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error as MAE


# In[2]:


# Create ANN model with quantile loss

class ANNRegressor_model():
    def __init__(self, units, num_features, quantile, file_dir,
               epochs, batch_size, deliverytime_scaler):
        self.units=units
        self.num_features = num_features
        self.quantile = quantile
        self.best_weights_dir = file_dir + 'ANNRegressor_{:.2f}quantile.h5py'.format(quantile)
        self.epochs = epochs
        self.batch_size = batch_size
        self.deliverytime_scaler = deliverytime_scaler
        self.wholenetwork_retrain_samplesize = 10000
        self.partnetwork_retrain_numlayer = 2
        self.create_model()

    # Pinball loss/ quantile score for multiple or single quantile
    def tilted_loss(self, y_true, y_pred):
        y_true = tf.keras.backend.cast(y_true, "float32")
        y_pred = tf.keras.backend.cast(y_pred, "float32")
        e = y_true - y_pred
        # find the average loss of quantile
        return tf.keras.backend.mean(tf.keras.backend.maximum(self.quantile * e, 
                                                      (self.quantile - 1) * e), axis = 0)

    def create_model(self):
        input_ = tf.keras.Input(shape=(self.num_features,))
        x = tf.keras.layers.Dense(self.units, activation='relu')(input_)
        x = tf.keras.layers.Dense(self.units, activation='relu')(x)
        x = tf.keras.layers.Dense(self.units, activation='relu')(x)
        output = tf.keras.layers.Dense(1, activation = "linear")(x)
        self.model = tf.keras.Model(inputs=[input_], outputs=[output])
        #self.model.summary()

    def fit_model(self, x_train, y_train):
        # Compile the model
        self.model.compile(loss=self.tilted_loss, optimizer='adam')
        
        # Set callbacks for training
        modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(self.best_weights_dir, 
                                                             monitor = 'val_loss',
                                                             save_best_only = True, 
                                                             save_weights_only = True,
                                                             verbose = 0)
        earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        callbacks = [modelcheckpoint, earlystopping]
        self.history = self.model.fit(x_train, y_train, callbacks = callbacks, 
                                      epochs = self.epochs, batch_size = self.batch_size, 
                                      verbose = 0, validation_split=0.15)
  
    def plot_result(self):
        plt.figure(figsize=(15, 5))
        plt.plot(self.history.history["loss"], "-*", label="training")
        plt.plot(self.history.history["val_loss"], "-o", label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("Penalized Loss")
        plt.legend()
        plt.show()
    
    # Predict based on the input features and inverse transform
    # Return rounded unscaled prediction time in days
    def predict(self, x_test):
        self.model.load_weights(self.best_weights_dir)
        y_pred = self.model.predict(x_test, batch_size=self.batch_size)
        return  np.round(self.deliverytime_scaler.inverse_transform(y_pred))

    def evaluate(self, x_test, y_test, plot=False):
        y_test_pred = self.predict(x_test) # rounded unscaled prediction time in days
        y_test = self.deliverytime_scaler.inverse_transform(y_test)

        if plot:
            # Plot prediction against actual
            plt.figure(figsize=(15, 5))
            plt.plot(y_test_pred, label='prediction')
            plt.plot(y_test, label='actual')
            plt.title('Prediction of Delivery Time (days) with {:.2f} quantile'.format(self.quantile))
            plt.ylabel('Delivery Time (days)')
            plt.xlabel('Samples')
            plt.legend()
            plt.show()

        # Evaluate the performance
        mae = MAE(y_test, y_test_pred)
        percentage_delayed_delivery = np.sum(y_test_pred < y_test) / len(y_test) * 100

        print ('The mean absolute error: {:.2f} days'.format(mae))
        print ('The percentage of delayed delivery: {:.2f}%'.format(percentage_delayed_delivery))

        return mae, percentage_delayed_delivery
    
    # Only set the last layer to be trainable when the dataset is small
    # Retrain the entire network when the dataset is large
    # Since the large dataset might contain much information and pattern
    # which is completely different from previous training set
    # Set trainable/ non-trainable before compile
    def refit_model(self, x_train, y_train):
        if len(x_train) < self.wholenetwork_retrain_samplesize:
            for i, layer in enumerate(self.model.layers[:self.partnetwork_retrain_numlayer]):
                layer.trainable = False
                print ('Layer {} set non-trainable'.format(i))
        
        self.model.compile(loss=self.tilted_loss, optimizer='adam')
        modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(self.best_weights_dir, 
                                                             monitor = 'val_loss',
                                                             save_best_only = True, 
                                                             save_weights_only = True,
                                                             verbose = 0)
        earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                         patience=20)
        callbacks = [modelcheckpoint, earlystopping]
        self.history = self.model.fit(x_train, y_train, callbacks = callbacks, 
                                      epochs = self.epochs, batch_size = self.batch_size, 
                                      verbose = 0, validation_split=0.15)

