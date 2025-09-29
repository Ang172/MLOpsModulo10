from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Crear una instancia de la aplicación Flask
app = Flask(__name__)

# Cargar modelo y lista de columnas correctas
# Se carga el archivo del modelo entrenado que contiene:
# - El modelo de machine learning
# - La lista de features en el orden correcto
model_data = joblib.load("model/model.joblib")
model = model_data["model"]  # Extraer el modelo entrenado
feature_names = model_data["features"]  # Extraer la lista de nombres de columnas

# Ruta principal - Endpoint de verificación de estado
@app.route("/", methods=["GET"])
def home():
    # Retorna un mensaje confirmando que la API está funcionando
    return jsonify({"message": "API funcionando correctamente"})

# Ruta de predicción - Endpoint principal del modelo
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Obtener los datos JSON enviados en la solicitud
        data = request.get_json()

        # Validar que los datos existen y contienen el campo 'input'
        if not data or "input" not in data:
            return jsonify({"error": "Se requiere 'input' con datos"}), 400
            # Código 400 = Bad Request (solicitud incorrecta)

        # Crear DataFrame asegurando orden correcto de columnas
        # Es CRUCIAL que las columnas estén en el mismo orden que durante el entrenamiento
        # [data["input"]]: Convierte la lista en una lista de una fila
        # columns=feature_names: Asegura el orden correcto de las features
        input_df = pd.DataFrame([data["input"]], columns=feature_names)

        # Hacer predicción
        # model.predict() retorna un array, tomamos el primer elemento [0]
        # ya que solo estamos prediciendo para un caso a la vez
        pred = model.predict(input_df)[0]
        
        # Retornar la predicción como JSON
        # int(pred): Convierte la predicción a entero (0 o 1)
        return jsonify({"prediction": int(pred)})
        # Código 200 = OK (éxito automático)

    except Exception as e:
        # Manejo de errores - captura cualquier excepción durante la predicción
        return jsonify({"error": str(e)}), 500
        # Código 500 = Internal Server Error (error del servidor)

# Punto de entrada principal de la aplicación
if __name__ == "__main__":
    # Ejecutar la aplicación Flask
    # host="0.0.0.0": Hace la app accesible desde cualquier IP
    # port=5000: Puerto donde escuchará las solicitudes
    app.run(host="0.0.0.0", port=5000)