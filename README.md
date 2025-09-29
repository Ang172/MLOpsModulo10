# Sistema de Clasificación de Cáncer de Mama con CI/CD

# Descripción
Sistema de Machine Learning para clasificación de tumores de cáncer de mama (benignos/malignos) implementado con Flask, Docker y pipeline CI/CD automatizado.

# Características
- **Modelo ML**: Random Forest con 96% de accuracy  
- **API REST**: Endpoints para predicciones en tiempo real  
- **Containerización**: Docker para entornos consistentes  
- **CI/CD**: Pipeline automatizado con GitHub Actions  
- **Pruebas**: Tests automatizados de endpoints y modelo  

# Tecnologías
- Python 3.10  
- Flask  
- Scikit-learn  
- Pandas  
- Docker  
- GitHub Actions  
- Joblib  

# Estructura del Proyecto
MLOpsModulo10/
- .github/workflows/
- ─ ci-cd.yml # Pipeline CI/CD
- model/
- ─ model.joblib # Modelo entrenado
- app.py # API Flask
- train_model.py # Entrenamiento del modelo
- data.csv # Dataset de entrenamiento
- Dockerfile # Configuración Docker
- requirements.txt # Dependencias

# Instalación y Ejecución

## Prerrequisitos
- Python 3.10+  
- Docker (opcional)  

# Método 1: Ejecución Local

## 1. Clonar repositorio
git clone https://github.com/Ang172/MLOpsModulo10.git
cd MLOpsModulo10

## 2. Instalar dependencias
pip install -r requirements.txt

## 3. Entrenar modelo (opcional)
python train_model.py

## 4. Ejecutar API
python app.py

#Método 2: Usando Docker

## 1. Construir imagen
docker build -t ml-cancer-classifier .

## 2. Ejecutar contenedor
docker run -d -p 5000:5000 --name ml-app ml-cancer-classifier

## 3. Verificar funcionamiento
curl http://localhost:5000/

##Uso de la API
Health Check

GET http://localhost:5000/

Respuesta

{
    "message": "API funcionando correctamente"
}

##Predicción

POST http://localhost:5000/predict
Content-Type: application/json

{
    "input": [
        17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 
        0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 
        153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 
        0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 
        0.7119, 0.2654, 0.4601, 0.1189
    ]
}


##Respuesta:

{
    "prediction": 1
}

0: Tumor Benigno (B)

1: Tumor Maligno (M)

