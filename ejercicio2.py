import requests
import pandas as pd


# Crear un DataFrame a partir de los datos
r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
renglones_iris = r.text.split('\n')


cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Convertir los datos en un DataFrame
df = pd.DataFrame([x.split(',') for x in renglones_iris[:-2]], columns=[cols])


statistics = df[cols].describe()
print(statistics)