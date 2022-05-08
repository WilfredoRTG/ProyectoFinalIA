from tensorflow import keras
from keras import layers, Sequential
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

size = 100
razas = []
imagenesProcesadas = []
indicesDeRazas = []

for index, tipos in enumerate(os.listdir('./dataSet')):
    razas.append(tipos)
    path = './dataSet/' + tipos
    for imagen in os.listdir(path):
        img_array = cv2.imread((path+'/'+imagen), cv2.IMREAD_GRAYSCALE)
        img_ajustadas = cv2.resize(img_array, (size, size))
        imagenesProcesadas.append(img_ajustadas)
        indicesDeRazas.append(index)

imagenesProcesadas = np.array(imagenesProcesadas).reshape(-1, size, size)
indicesDeRazas = np.array(indicesDeRazas).reshape(-1)

model = Sequential([
    layers.Flatten(input_shape=(size, size)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)
])

myLossFunction = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(
    optimizer='adam',
    loss=myLossFunction,
    metrics=['accuracy']
)

historial = model.fit(imagenesProcesadas, indicesDeRazas, epochs=20, verbose=False)
# plt.xlabel('#Epoca')
# plt.ylabel('Perdida')
# plt.plot(historial.history['loss'])
# plt.show()
test_loss, test_acc = model.evaluate(imagenesProcesadas, indicesDeRazas, verbose=2)

print('\nTest:', test_acc)
probability_model = keras.Sequential([
    model, keras.layers.Softmax()
])
predictions = probability_model.predict(imagenesProcesadas)
print("\nIntento de prediccion a ", razas[indicesDeRazas[0]])
razaPre = razas[np.argmax(predictions[0])]

if(razaPre==razas[indicesDeRazas[0]]):
    print("La prediccion es correcta")
    print('Resultado de la prediccion: ', razaPre)
else:
    print("La prediccion es incorrecta")
    print("Resultado de la prediccion: ", razaPre)
