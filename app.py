from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar modelo y lista de columnas correctas
model_data = joblib.load("model/model.joblib")
model = model_data["model"]
feature_names = model_data["features"]

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API funcionando correctamente"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "input" not in data:
            return jsonify({"error": "Se requiere 'input' con datos"}), 400

        # Crear DataFrame asegurando orden correcto de columnas
        input_df = pd.DataFrame([data["input"]], columns=feature_names)

        # Hacer predicción
        pred = model.predict(input_df)[0]
        return jsonify({"prediction": int(pred)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
