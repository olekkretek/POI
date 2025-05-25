import numpy as np
from tensorflow.keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


#Wczytywanie danych z pliku csv
csv_filepath = 'features.csv'
df = pd.read_csv(csv_filepath)

#Wyodrębnianie cech do macierzy x oraz etykiet do wektora y
x = df.iloc[:, 2:].values
y = df.iloc[:, 0].values
print(x)
print(y)

#Wstępne przetwarzanie danych
label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(y).reshape(-1, 1)

onehot_encoder = OneHotEncoder(sparse_output=False)
y_onehot = onehot_encoder.fit_transform(y_int)

X_train, X_test, y_train_onehot, y_test_onehot = train_test_split(
    x,
    y_onehot,
    test_size=0.3,    # 30% danych na zbiór testowy
)
#Tworzenie modelu sieci neuronowej
model = Sequential()
input_dimension = X_train.shape[1]
model.add(Dense(units=10, activation='sigmoid', input_shape=(input_dimension,)))
model.add(Dense(units = 4, activation='softmax'))
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#Uczenie sieci
model.fit(
    X_train,
    y_train_onehot,
    epochs=100,
    batch_size=10,
    shuffle = True
)
#Testowanie sieci
y_pred = model.predict(X_test)
y_pred_int = np.argmax(y_pred, axis=1)
y_test_int = np.argmax(y_test_onehot, axis=1)
cm = confusion_matrix(y_test_int, y_pred_int)
print(cm)