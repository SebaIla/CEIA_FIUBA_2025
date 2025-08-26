# Importa librerías necesarias para hacer peticiones HTTP, manejar JSON, controlar tiempos, manipular datos y rutas de archivos
import requests
import json
import time
import pandas as pd
from pathlib import Path

# Función principal para obtener datos de colisiones desde la API de Los Ángeles
def fetch_collision_data():
    # URL del endpoint de la API pública
    api_ep_la_collisions = 'https://data.lacity.org/resource/d5tf-ez2w.json'
    
    # Parámetros de configuración para la descarga
    LIMIT = 50000               # Número máximo de registros por petición
    MAX_RETRIES = 3             # Número máximo de reintentos en caso de error
    RETRY_DELAY = 10            # Tiempo de espera entre reintentos (segundos)
    EXPECTED_TOTAL_ROWS = 621677  # Número esperado de registros totales

    # Inicializa variables para almacenar los datos y controlar el progreso
    la_collisions_data = []
    offset = 0
    total_retrieved = 0

    print("Starting data retrieval...")  # Mensaje inicial

    # Bucle principal para descargar los datos en bloques
    while True:
        retries = 0
        success = False
        last_error_message = ""

        # Bucle de reintentos en caso de fallos
        while retries < MAX_RETRIES and not success:
            try:
                # Define los parámetros de la petición (limit y offset)
                params = {'$limit': LIMIT, '$offset': offset}
                response = requests.get(api_ep_la_collisions, params=params)

                # Si la respuesta es exitosa (HTTP 200)
                if response.status_code == 200:
                    chunk = response.json()  # Convierte la respuesta en JSON
                    la_collisions_data.extend(chunk)  # Agrega los datos al acumulador
                    retrieved = len(chunk)
                    total_retrieved += retrieved
                    print(f"✅ Partially ingested {retrieved} rows (Total: {total_retrieved})")
                    offset += retrieved  # Actualiza el offset para la siguiente petición
                    success = True
                else:
                    # Si hay error HTTP, muestra mensaje y espera para reintentar
                    last_error_message = f"HTTP Error {response.status_code}: {response.text}"
                    print(f"⚠️ {last_error_message}. Retrying...")
                    retries += 1
                    if retries < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
            except Exception as e:
                # Si ocurre una excepción (por ejemplo, error de red)
                last_error_message = f"Exception: {str(e)}"
                print(f"⚠️ {last_error_message}. Retrying...")
                retries += 1
                if retries < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)

        # Si no se logró obtener datos después de varios intentos, se detiene
        if not success:
            print(f"❌ Failed after {MAX_RETRIES} attempts. Stopping. Last error: {last_error_message}")
            break

        # Si el bloque recibido tiene menos registros que el límite, significa que ya no hay más datos
        if len(chunk) < LIMIT:
            break

    # Mensaje final con el total de registros obtenidos
    print(f"✅ Final ingestion completed: {len(la_collisions_data)} rows retrieved.")

    # Verifica si se obtuvo el número esperado de registros
    if len(la_collisions_data) == EXPECTED_TOTAL_ROWS:
        print(f"🎉 Successfully retrieved all {EXPECTED_TOTAL_ROWS} expected rows.")
    else:
        print(f"❗ Warning: Expected {EXPECTED_TOTAL_ROWS} rows, but retrieved {len(la_collisions_data)} rows.")
    
    return la_collisions_data  # Devuelve los datos obtenidos

# Función para guardar los datos en un archivo JSON
def save_raw_data(data, output_path):
    # Crea el directorio si no existe
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Guarda los datos en formato JSON
    with open(output_path, 'w') as f:
        json.dump(data, f)
    
    print(f"Data saved to {output_path}")  # Mensaje de confirmación

# Punto de entrada del script
if __name__ == "__main__":
    data = fetch_collision_data()  # Obtiene los datos
    save_raw_data(data, "data/raw/la_collisions_raw.json")  # Los guarda en disco