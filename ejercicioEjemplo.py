import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt


# Obtenemos la data de iris.data
r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')

# Creando una lista a partir de una separación de la cadena
renglones_iris = r.text.split('\n')

# Verificamos los datos de iris.data en las primeras 5 posiciones

#print(renglones_iris[:5])
# print([ renglon.split(',')[:-1] for renglon in renglones_iris[:-2]])
# print(map(float, renglon.split(',')[:-1]) for renglon in   renglones_iris[:-2])


# Nuevo arreglo
iris_data = [list(map(float, renglon.split(',')[:-1])) for renglon in renglones_iris[:-2]]
iris = np.array(iris_data)

# print(iris[:10])

# definimos x y y, indicamos que lo queremos desde el indice 0 hasta 50
x = iris[:50,0]
y = iris[:50,1]


# asignamos los valores que queremos ver en la grafica
plt.plot(x, y, 'r.')


plt.plot(iris[:50,0], iris[:50,1], 'r.')

plt.plot(iris[51:100,0], iris[51:100,1], 'b.')


plt.plot(iris[101:,0], iris[101:,1], 'g.')

# mostramos la grafica
plt.show()


