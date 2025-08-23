# Utilizando docker compose

## 1.Estructura de archivos

proyecto/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── run.sh
├── src/
│   ├── etl.py
│   ├── train.py
│   └── test.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
└── results/

## 2.Dockerfile

FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

RUN mkdir -p data/raw data/processed models results

VOLUME ["/app/data", "/app/models", "/app/results"]

CMD ["python", "src/etl.py"]

## 3.requirments.txt

requests==2.31.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.2

## 4.Docker-compose.yml

version: '3.8'

services:
  # Servicio para el proceso ETL
  etl:
    build: .
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    command: python src/etl.py
    restart: on-failure

  # Servicio para entrenamiento del modelo
  train:
    build: .
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    command: python src/train.py
    restart: on-failure
    depends_on:
      - etl

  # Servicio para testing del modelo
  test:
    build: .
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./results:/app/results
      - ./logs:/app/logs
    command: python src/test.py
    restart: on-failure
    depends_on:
      - train

  # Servicio para ejecutar todo el pipeline
  pipeline:
    build: .
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./results:/app/results
      - ./logs:/app/logs
    command: sh -c "python src/etl.py && python src/train.py && python src/test.py"
    restart: on-failure

volumes:
  data:
  models:
  results:
  logs:

# 6.run.sh (script de ejecución)

#!/bin/bash

# Script para ejecutar el pipeline con Docker Compose
echo "Iniciando el pipeline de colisiones de tráfico..."

# Crear directorios si no existen
mkdir -p data/raw data/processed models results logs

# Construir los servicios
echo "Construyendo la imagen Docker..."
docker-compose build

# Ejecutar todo el pipeline
echo "Ejecutando el pipeline completo..."
docker-compose up pipeline

# Alternativa: ejecutar servicios individualmente
# echo "Ejecutando ETL..."
# docker-compose up etl
#
# echo "Ejecutando entrenamiento..."
# docker-compose up train
#
# echo "Ejecutando testing..."
# docker-compose up test

echo "Pipeline completado. Verifique los resultados en las carpetas data/, models/ y results/"

# 7.Proceso de ejecución VS CODE

1. En la terminal de VS CODE

2. Dar permisos de ejecución al script

chmod +x run.sh

3. Ejecutar el pipeline completo

./run.sh

4. O ejecutar servicios individualmente

# Solo ETL
docker-compose up etl

# Solo entrenamiento (después de ETL)
docker-compose up train

# Solo testing (después de entrenamiento)
docker-compose up test

5. Ver logs de un servicio en especifico

docker-compose logs etl
docker-compose logs train
docker-compose logs test

6. Detener servicios

docker-compose down
