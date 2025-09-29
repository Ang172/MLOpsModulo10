import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

CSV_FILE = "data.csv"
os.makedirs("model", exist_ok=True)

# Leer CSV
data = pd.read_csv(CSV_FILE)

# Eliminar columnas innecesarias
for col in ['id', 'Unnamed: 32']:
    if col in data.columns:
        data = data.drop(columns=[col])

# Convertir diagnosis a 1/0
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Features y target
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']

print(f"Número de columnas después de limpiar: {len(X.columns)}")
print(f"Columnas finales: {X.columns.tolist()}")

# Split datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Guardar modelo
joblib.dump({"model": model, "features": X.columns.tolist()}, "model/model.joblib")
print("Modelo guardado en model/model.joblib")
