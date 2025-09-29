import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Configuración de archivos y directorios
CSV_FILE = "data.csv"  # Archivo de datos de entrada
os.makedirs("model", exist_ok=True)  # Crear directorio para guardar el modelo si no existe

# Leer CSV - Carga el dataset desde el archivo CSV
data = pd.read_csv(CSV_FILE)

# Eliminar columnas innecesarias
# Remover columnas que no son útiles para el modelo (ID y columnas vacías)
for col in ['id', 'Unnamed: 32']:
    if col in data.columns:
        data = data.drop(columns=[col])

# Convertir diagnosis a 1/0
# Transformar la variable objetivo de texto (M/B) a numérico (1/0)
# M = Maligno (1), B = Benigno (0)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Features y target
# X: Variables independientes (todas las columnas excepto diagnosis)
# y: Variable dependiente (diagnosis - lo que queremos predecir)
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']

# Información de depuración - mostrar cuántas y cuáles columnas usaremos
print(f"Número de columnas después de limpiar: {len(X.columns)}")
print(f"Columnas finales: {X.columns.tolist()}")

# Split datos
# Dividir el dataset en entrenamiento (80%) y prueba (20%)
# random_state=42 para reproducibilidad
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo
# Crear y entrenar un modelo de Random Forest (bosque aleatorio)
# n_estimators=100: 100 árboles en el bosque
# random_state=42: Semilla para reproducibilidad
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  # Entrenar el modelo con datos de entrenamiento

# Evaluación
# Predecir con datos de prueba y calcular precisión
y_pred = model.predict(X_test)  # Hacer predicciones en el conjunto de prueba
accuracy = accuracy_score(y_test, y_pred)  # Calcular porcentaje de aciertos
print(f"Accuracy: {accuracy:.2f}")  # Mostrar precisión (ej: 0.96 = 96%)

# Guardar modelo
# Guardar tanto el modelo entrenado como la lista de features usadas
# Esto es importante para que la app sepa en qué orden esperar los datos
joblib.dump({"model": model, "features": X.columns.tolist()}, "model/model.joblib")
print("Modelo guardado en model/model.joblib")  # Confirmación de guardado exitoso