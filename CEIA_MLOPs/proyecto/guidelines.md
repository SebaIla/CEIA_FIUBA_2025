Estructura de carpetas

proyecto/
├── Dockerfile
├── requirements.txt
├── src/
│   ├── etl.py
│   ├── train.py
│   └── test.py
└── data/
    ├── raw/
    └── processed/

# DOCKERFILE

## Usar una imagen base de Python oficial
FROM python:3.9-slim

## Establecer el directorio de trabajo
WORKDIR /app

## Copiar los archivos de requisitos primero (para aprovechar el cache de Docker)
COPY requirements.txt .

## Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

## Copiar el código fuente
COPY src/ ./src/
COPY data/ ./data/

## Crear directorios para datos si no existen
RUN mkdir -p data/raw data/processed

## Establecer el punto de entrada por defecto
CMD ["python", "src/etl.py"]

# requirements.txt

requests==2.31.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.2

# Comandos para construir y ejecutar

## Construir la imagen
docker build -t collision-prediction .

## Ejecutar el ETL
docker run -v $(pwd)/data:/app/data collision-prediction python src/etl.py

## Ejecutar el entrenamiento
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models collision-prediction python src/train.py

## Ejecutar las pruebas
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models -v $(pwd)/results:/app/results collision-prediction python src/test.py

## Ejecutar todo en secuencia
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models -v $(pwd)/results:/app/results collision-prediction sh -c "python src/etl.py && python src/train.py && python src/test.py"