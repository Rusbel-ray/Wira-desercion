from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import os

# Inicializar la aplicación
app = Flask(__name__)

# Cargar el modelo entrenado
with open('modelo_rf.pkl', 'rb') as f:
    model = pickle.load(f)

# Ruta principal para mostrar el formulario
@app.route('/')
def home():
    return render_template('form.html')

# Ruta para hacer predicciones
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Obtener datos JSON de la solicitud
    df = pd.DataFrame(data, index=[0])  # Convertir a DataFrame
    prediction = model.predict(df)  # Hacer la predicción
    return jsonify({'predicciones': prediction.tolist()})  # Devolver la predicción como JSON

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
