
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import numpy as np

class lstm:
    
    def __init__(self, X_train, y_train):

        # Assuming X_train and y_train are already preprocessed sequences
        # X_train shape: (number of sequences, number of time steps, number of features)
        self.x = X_train
        # y_train shape: (number of sequences,)
        self.y = y_train

        # LSTM model
        self.model = Sequential()
        # Adjust input_shape to match the structure of your data
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        self.model.add(Dropout(rate=0.2))

        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(Dropout(rate=0.2))

        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(Dropout(rate=0.2))

        self.model.add(LSTM(units=50, return_sequences=False))
        self.model.add(Dropout(rate=0.2))

        # Output layer with 1 unit for regression
        self.model.add(Dense(units=1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        print("Architecture of the Multi-Stack LSTM Layer is ready!")

    def train_it(self):
        self.model.fit(self.x, self.y, batch_size=32, epochs=20)
        return self.model

    def predict_it(self, data):
        # Assuming data is a preprocessed sequence for prediction
        lst = self.model.predict(data)
        return lst
    
