# Importa módulos estándar para manejo de sistema, ejecución de scripts y rutas
import os
import sys
import subprocess
from pathlib import Path

# Desactiva plugins de Metaflow que pueden causar problemas de compatibilidad en Windows
os.environ['METAFLOW_DISABLE_PLUGINS'] = 'kubernetes,aws,batch,azure'

# Intenta importar Metaflow; si no está instalado, muestra un mensaje y termina el programa
try:
    from metaflow import FlowSpec, step
except ImportError as e:
    print(f"Metaflow import error: {e}")
    print("Please install Metaflow: pip install metaflow")
    sys.exit(1)

# Define una clase que representa el flujo de trabajo para predicción de colisiones
class CollisionPredictionFlow(FlowSpec):
    
    @step
    def start(self):
        """Paso inicial del flujo"""
        print("Starting collision prediction workflow")
        self.next(self.ingest_data)  # Avanza al paso de ingestión de datos

    @step
    def ingest_data(self):
        """Paso que ejecuta la ingestión de datos desde la API de Los Ángeles"""
        print("Ingesting data from LA API...")
        try:
            # Usa el mismo intérprete de Python que ejecuta este script
            python_executable = sys.executable
            # Ejecuta el script de ingestión como subproceso
            result = subprocess.run(
                [python_executable, 'src/data_ingestion.py'], 
                capture_output=True, 
                text=True, 
                cwd=os.getcwd()
            )
            # Muestra la salida estándar del script
            print(result.stdout)
            # Si hay errores, los muestra también
            if result.stderr:
                print(f"Stderr: {result.stderr}")
            # Si el script falla, informa el código de error
            if result.returncode != 0:
                print(f"Data ingestion failed with return code: {result.returncode}")
                # Continúa de todos modos (útil para demostraciones)
        except Exception as e:
            print(f"Error in data ingestion: {e}")
        self.next(self.process_data)  # Avanza al paso de procesamiento

    @step
    def process_data(self):
        """Paso que ejecuta el procesamiento de datos"""
        print("Processing data...")
        try:
            python_executable = sys.executable
            result = subprocess.run(
                [python_executable, 'src/data_processing.py'], 
                capture_output=True, 
                text=True, 
                cwd=os.getcwd()
            )
            print(result.stdout)
            if result.stderr:
                print(f"Stderr: {result.stderr}")
            if result.returncode != 0:
                print(f"Data processing failed with return code: {result.returncode}")
        except Exception as e:
            print(f"Error in data processing: {e}")
        self.next(self.train_model)  # Avanza al paso de entrenamiento

    @step
    def train_model(self):
        """Paso que ejecuta el entrenamiento del modelo"""
        print("Training model...")
        try:
            python_executable = sys.executable
            result = subprocess.run(
                [python_executable, 'src/train.py'], 
                capture_output=True, 
                text=True, 
                cwd=os.getcwd()
            )
            print(result.stdout)
            if result.stderr:
                print(f"Stderr: {result.stderr}")
            if result.returncode != 0:
                print(f"Model training failed with return code: {result.returncode}")
        except Exception as e:
            print(f"Error in model training: {e}")
        self.next(self.end)  # Avanza al paso final

    @step
    def end(self):
        """Paso final que indica que el flujo ha terminado"""
        print("Workflow completed!")

# Punto de entrada del script
if __name__ == '__main__':
    # Agrega el directorio actual al path de Python para facilitar importaciones
    sys.path.insert(0, os.getcwd())
    
    # Ejecuta el flujo de trabajo
    CollisionPredictionFlow()
