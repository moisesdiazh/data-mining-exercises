import matplotlib.pyplot as plt
import requests
import pandas as pd

# Crear un DataFrame a partir de los datos
r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
renglones_iris = r.text.split('\n')


cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Convertir los datos en un DataFrame
df = pd.DataFrame([x.split(',') for x in renglones_iris[:-2]], columns=[cols])

x = df['sepal_length'].values.tolist()

y = df['sepal_width'].values.tolist()

plt.scatter(x, y)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Scatter Plot of Sepal Length vs. Sepal Width')

plt.show()