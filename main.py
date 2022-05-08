# Wilfredo Rafael Tablero Gomez 166918

# Primero tengo que mencionar que esta red neuronal lo que hace es predecir una raza de perro, esto se hizo haciendo
# el propio dataset con 5 diferentes razas de perro, Golden Doodle, Golden Retriever, Maltese, Pastor Aleman y Schnauzer
# finalmente, el modelo te arrojara una prediccion para la primera raza de perro (Golden Doodle), y se evaluara si la 
# prediccion es correcta.

# Iniciamos importando las librerias necesarias, en este caso tensorflow, keras, matplotlib
# pero esta es solo para desmostraciones de las graficas de perdida, despues open cv, esta libreria
# solamente es para convertir las imágenes del dataset a escala de grises y tambien para redimensionarla,
# despues importe la libreria de numpy, esta para poder manejar los tipos de arreglos que maneja los metodos
# de tensorflow, por ultimo importe la libreria de os, esta para poder acceder a las carpetas y archivos

from tensorflow import keras
from keras import layers, Sequential
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

size = 100  #Esto es para redimenzionar las imágenes y tambien cambiar el tamaño del arreglo 
razas = []  #Este arreglo es para almacenar todas las razas de perros que tenemos
imagenesProcesadas = [] #Este arreglo es para guardar todas las imágenes ya ajustadas y convertidas
indicesDeRazas = [] #Este arreglo es para guardar los índices de las razas, solo guarda numeros entre 0 a 4

# Para empezar a obtener todas las carpetas e imágenes, usamos el comando listdir y dando como parametro la carpeta
# donde esta todo el dataSet, se puede acceder a todas las carpetas que este contiene, que en este caso son todas las razas
# de perros que contiene el dataSet. 
for index, tipos in enumerate(os.listdir('./dataSet')):
    # Se agrega todas las razas de perros al arreglo, ya que el nombre de las carpetas es tal cual en nombre de las razas.
    razas.append(tipos)
    # Despues a una variable se asigan el path para cada una de las carpetas, para despues extraer todas las imágenes de 
    # cada una de las carpetas.
    path = './dataSet/' + tipos
    for imagen in os.listdir(path):
        # Ya que las imágenes tiene tres canales de colores (RGB), se necesita convertir todas las imágenes a uno de solo 1 canal,
        # para hacer esto usando la libreria cv2 (open cv), se leen las imágenes y se convierten a escala de grises usando IMREAD_GRAYSCALE.
        img_array = cv2.imread((path+'/'+imagen), cv2.IMREAD_GRAYSCALE)
        # Despues de tener las imágenes en solo un canal de colores, se necesita redimenzionar las imágenes, esto para tener un mismo
        # tamaño para cada una de ellas y poder trabajar de una mejor manera
        img_ajustadas = cv2.resize(img_array, (size, size))
        # Se agregan el arreglo de las imágenes redimenzionadas y con escala de grises
        imagenesProcesadas.append(img_ajustadas)
        # Se agregan los índices de cada imagen
        indicesDeRazas.append(index)

# Estos arreglos se hace un arreglo de numpy para poder utilizarlos en las funciones de tensorflow, ya que este solo permite
# el tipo de arreglo numpy.ndarray, de igual manera al arreglo de imágenes se tiene que dar una forma de 100*100 ya que las
# imágenes son de este tamaño, por otro lado, a los índices solo se hace una forma de 1
imagenesProcesadas = np.array(imagenesProcesadas).reshape(-1, size, size)
indicesDeRazas = np.array(indicesDeRazas).reshape(-1)

# Se necesita usar un modelo de keras, en este caso se esta utilizando el modelo Secuencial, utilizando el método Flatten,
# se transforma los arreglos bidimensionales a un arreglo de una sola dimensión, despues se declara el numero de neuronas
# usando el método Dense, en este caso declare que quiero 120 neuronas, por ultimo se definen el numero de capas de salida, 
# el cual este es una arreglo de tamaño de 10
model = Sequential([
    layers.Flatten(input_shape=(size, size)),
    layers.Dense(120, activation='relu'),
    layers.Dense(10)
])

# Esta función de perdida nos ayuda a medir que tan exacto es el modelo durante el entrenamiento
myLossFunction = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Para compilar el modelo primero se necesita usar un optimizador, esto es para que la red neuronal sepa como ajustar los pesos,
# en este caso se esta utilizando el algoritmo Adam, el siguiente parametro es la función de perdida que se declaro anteriormente,
# por ultimo se declara que tipo de metrica se requiere, en este caso se escogio accuracy, esta lo que hace es calcular que tan
# a menudo las predicciones son igual a las labels (indicesDeRazas)
model.compile(
    optimizer='adam',
    loss=myLossFunction,
    metrics=['accuracy']
)

# Para el entrenamiento del modelo, se necesitan las imágenes, los índices y el numero de vueltas, en este caso se llama el
# arreglo de las imágenes ya redimenzionadas y en escala de grises, y los índices de las razas, despues se define el numero de
# vueltas (epochs), en este caso se definio 40 vueltas, pero este numero se puede aumentar, aun que es importante mencionar, que
# llega un punto donde muchisimas vueltas, la red ya no sigue aprendiendo, por lo tanto estamos malgastando recursos, el ultimo
# parametro (verbose), solo es para que no imprima los resultados de cada una de las vueltas.
historial = model.fit(imagenesProcesadas, indicesDeRazas, epochs=40, verbose=False)

# En esta gráfica se puede ver como la red va aprendiendo durante todas las vueltas que va haciendo, como se menciono antes,
# se puede apreciar un punto donde la red ya no esta aprendiendo, pero este no es mucho.
plt.xlabel('#Epoca')
plt.ylabel('Perdida')
plt.plot(historial.history['loss'])
plt.show()

# Se hace un test para probar que tan bien esta entrenado el modelo, para esto se usa el método evaluate simplemente pasando
# el arreglo de imágenes y sus índices, asi como cuando entrenamos el modelo verbose se declara como falso para evitar las impresiones
test_loss, test_acc = model.evaluate(imagenesProcesadas, indicesDeRazas, verbose=False)
print('\nTest:', test_acc)

# Ya con el modelo entrenado, este se puede usar para hacer predicciones, utilizando el metodo Softmax para convertir los datos 
# de salida del modelo a probabilidades
probability_model = Sequential([
    model, layers.Softmax()
])

# Ahora se realiza las predicciones
predicciones = probability_model.predict(imagenesProcesadas)
print("\nIntento de prediccion a primera imagen de Golden Doodle")

# Se esta haciendo una predicción para la primera imagen la cual corresponde a la raza de perro Golden Doodle, utilizando 
# np.argmax podemos obtener el numero mas alto, en otras palabras la predicción con la cual el modelo tiene mas confianza
razaPre = razas[np.argmax(predicciones[0])]

# Se hace una comparacion para ver si la prediccion es correcta
if(razaPre=='Golden Doodle'):
    print("La prediccion es correcta")
    print('Resultado de la prediccion: ', razaPre)
else:
    print("La prediccion es incorrecta")
    print("Resultado de la prediccion: ", razaPre)
