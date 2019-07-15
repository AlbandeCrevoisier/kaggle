""" Kaggle - Digit Recognizer - MNIST """
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.utils import to_categorical


dataset = pd.read_csv("train.csv")

y = to_categorical(dataset.pop('label'))
X = np.array(dataset).reshape(-1, 28, 28, 1) / 255
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

model = Sequential()
model.add(Conv2D(64, 3, padding='same', activation='relu',
    input_shape=(28, 28, 1)))
model.add(Conv2D(64, 3, activation='relu'))
model.add(MaxPooling2D(2))
model.add(Dropout(0.05))

model.add(Conv2D(128, 3, padding='same', activation='relu'))
model.add(Conv2D(128, 3, activation='relu'))
model.add(MaxPooling2D(2))
model.add(Dropout(0.05))

model.add(Flatten())
model.add(Dense(28, 'relu'))
model.add(Dropout(0.1))
model.add(Dense(10, 'softmax'))

model.compile(loss='categorical_crossentropy',
    optimizer=Adadelta(),
    metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test))

#test = pd.read_csv("test.csv")
#p = pd.DataFrame({'ImageId': test.index + 1, 'Label': clf.predict(test)})
#p.to_csv("sub.csv", index=False)
