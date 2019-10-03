""" Kaggle - Digit Recognizer - MNIST """
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical


dataset = pd.read_csv("train.csv")

y = to_categorical(dataset.pop('label'))
X = np.array(dataset).reshape(-1, 28, 28, 1) / 255
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(Conv2D(128, 3, padding='same', activation='relu',
    input_shape=(28, 28, 1)))
model.add(MaxPooling2D(2))
model.add(Conv2D(128, 3, activation='relu'))
model.add(Conv2D(128, 3, activation='relu'))

model.add(Flatten())
model.add(Dropout(0.1))
model.add(Dense(128, 'relu'))
model.add(Dense(256, 'relu'))
model.add(Dense(128, 'relu'))
model.add(Dense(10, 'softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test))

test = pd.read_csv("test.csv")
test = np.array(test).reshape(-1, 28, 28, 1) / 255
y_pred = model.predict(test).argmax(axis=1)
p = pd.DataFrame({'ImageId': np.arange(1, len(test) + 1), 'Label': y_pred})
p.to_csv("sub.csv", index=False)
