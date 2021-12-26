# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pandas as pd

train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(
    fname=os.path.basename(train_dataset_url),
    origin=train_dataset_url
)
print("Local copy of the dataset file: {}".format(train_dataset_fp))

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
dataset = pd.read_csv(train_dataset_url, names=column_names, skiprows=1)

X = dataset.iloc[:, 0:-1]
Y = dataset.iloc[:, -1]

plt.scatter(X.iloc[:, 2], X.iloc[:, 0], c=Y, cmap='viridis')
plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.show()

Y = Y.values.reshape(-1, 1)

ec = OneHotEncoder(sparse=False)
Y = ec.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=64, activation='relu'))
ann.add(tf.keras.layers.Dense(units=64, activation='relu'))
ann.add(tf.keras.layers.Dense(units=3, activation='softmax'))
ann.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
ann.fit(X_train, Y_train, batch_size=32, epochs=50)

results = ann.evaluate(X_test, Y_test, batch_size=32)
print("test loss, test acc:", results)
