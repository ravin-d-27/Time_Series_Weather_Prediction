
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import numpy as np

class lstm:
    
    def __init__(self, X_train, y_train, n_cols):

        # Assuming X_train and y_train are already preprocessed sequences
        # X_train shape: (number of sequences, number of time steps, number of features)
        self.x = X_train
        # y_train shape: (number of sequences,)
        self.y = y_train

        self.model = Sequential([
                LSTM(50, return_sequences= True, input_shape= (X_train.shape[1], n_cols)),
                LSTM(64, return_sequences= False),
                Dense(32),
                Dense(16),
                Dense(n_cols)
            ])

        print("Architecture of the Multi-Stack LSTM Layer is ready!")
        self.model.compile(optimizer='adam', loss='mse', metrics="mean_absolute_error")
        self.model.summary()

    def train_it(self):
        history = self.model.fit(self.x, self.y, batch_size=32, epochs=100)
        return self.model, history

    def predict_it(self, data):
        # Assuming data is a preprocessed sequence for prediction
        lst = self.model.predict(data)
        return lst
    
