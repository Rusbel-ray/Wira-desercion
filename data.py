import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Cargar el dataset
data = pd.read_csv('cleaned_dataset.csv')

# Separar variables independientes (X) y la variable dependiente (y)
X = data.drop(columns=['Desercion'])
y = data['Desercion']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar el modelo
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Guardar el modelo
with open('modelo_rf.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Modelo guardado como modelo_rf.pkl")
