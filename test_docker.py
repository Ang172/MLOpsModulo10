import requests
import time
import subprocess
import sys

def test_docker_container():
    print(" Iniciando tests del contenedor Docker...")
    
    try:
        # Construir la imagen
        print("1. Construyendo imagen Docker...")
        build_result = subprocess.run(
            ["docker", "build", "-t", "ml-app", "."],
            capture_output=True,
            text=True
        )
        
        if build_result.returncode != 0:
            print(f" Error construyendo imagen: {build_result.stderr}")
            return False
        
        print(" Imagen construida exitosamente")
        
        # Ejecutar contenedor
        print("2. Iniciando contenedor...")
        run_result = subprocess.run(
            ["docker", "run", "-d", "-p", "5000:5000", "--name", "ml-test", "ml-app"],
            capture_output=True,
            text=True
        )
        
        if run_result.returncode != 0:
            print(f" Error iniciando contenedor: {run_result.stderr}")
            return False
        
        print(" Contenedor iniciado")
        
        # Esperar que la app esté lista
        print("3. Esperando que la app esté lista...")
        time.sleep(8)
        
        # Test endpoint home
        print("4. Testeando endpoint home...")
        try:
            response = requests.get("http://localhost:5000/", timeout=10)
            if response.status_code == 200:
                print(" Home endpoint funciona")
                print(f"   Respuesta: {response.json()}")
            else:
                print(f" Home endpoint error: {response.status_code}")
                return False
        except Exception as e:
            print(f" Error en home endpoint: {e}")
            return False
        
        # Test endpoint predict
        print("5. Testeando endpoint predict...")
        sample_data = {
            "input": [
                17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 
                0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 
                153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 
                0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 
                0.7119, 0.2654, 0.4601, 0.1189
            ]
        }
        
        try:
            response = requests.post(
                "http://localhost:5000/predict",
                json=sample_data,
                timeout=10
            )
            
            if response.status_code == 200:
                prediction = response.json()
                print(" Predict endpoint funciona")
                print(f"   Predicción: {prediction}")
            else:
                print(f" Predict endpoint error: {response.status_code}")
                print(f"   Respuesta: {response.text}")
                return False
                
        except Exception as e:
            print(f" Error en predict endpoint: {e}")
            return False
        
        print(" ¡Todos los tests pasaron! El Docker funciona correctamente")
        return True
        
    finally:
        # Limpiar siempre
        print("6. Limpiando contenedor...")
        subprocess.run(["docker", "stop", "ml-test"], capture_output=True)
        subprocess.run(["docker", "rm", "ml-test"], capture_output=True)
        print(" Contenedor limpiado")

if __name__ == "__main__":
    success = test_docker_container()
    sys.exit(0 if success else 1)