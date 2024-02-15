import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import Accuracy

# Load data from CSV file
data = pd.read_csv('./data.csv')

# Extract features (8x8 pixels)
X = data.iloc[:, :64].values.reshape(-1, 8, 8, 1)

# Extract labels (last column)
y = data.iloc[:, 64].values.reshape(-1, 1)

# One-hot encode the labels
encoder = OneHotEncoder(categories='auto')
y = encoder.fit_transform(y).toarray()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the convolutional neural network
model = Sequential([
    Input(shape=(8, 8, 1)),
    Conv2D(5, (5, 5), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(4, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=[Accuracy()])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=30, validation_data=(X_test, y_test))