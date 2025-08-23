# ESPECIALIZACIÓN EN INTELIGENCIA ARTIFICAL
## OPERACIONES DE APRENDIZAJE AUTOMÁTICO I
### TRABAJO FINAL
#### Integrantes
    1. Corti, Gaston
    2. Castillo, Sebastian

# MODELO IMPLEMENTADO EN LOCAL UTILIZANDO MLFLOW Y METAFLOW

Este repositorio implementa un ciclo completo de desarrollo de machine learning local con:

    Gestión de Ambiente: Conda para dependencias y reproducibilidad

    Orquestación: Metaflow para gestionar el flujo de trabajo de datos

    Seguimiento de Experimentos: MLflow para registrar modelos y métricas

    Modularidad: Código organizado en componentes reutilizables

    Control de Versiones: Git con archivos ignorados apropiados para datos y modelos

El flujo automatizado incluye ingestión de datos desde la API de LAPD, preprocesamiento, entrenamiento de un modelo Random Forest y registro del modelo con MLflow. Los usuarios pueden ejecutar el pipeline completo con un solo comando y visualizar los resultados mediante la interfaz de MLflow.

## Herramientas

    1. Ambiente creado en conda en windows con los siguientes dependencias

    - dependencies:
    - python=3.11.13
    - jupyter
    - pandas
    - numpy
    - scikit-learn
    - matplotlib
    - requests
    - pip
    - pip:
        - mlflow
        - metaflow

## Python
    
    1. Version de Python : 3.11.13

## Estructura de las carpetas

collision-model-project/
├── environment.yml
├── README.md
├── data/
│   ├── raw/
│   ├── processed/
├── notebooks/
│   └── collision_model.ipynb
├── src/
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── train.py
├── flows/
│   └── collision_flow.py
├── models/
│   └── (modelos guardados por MLflow)
└── mlruns/
    └── (experimentos de MLflow)

## Instalación

1. Clonar el repositorio.
2. Crear el ambiente conda: `conda env create -f environment.yml`
3. Activar el ambiente: `conda activate collision-model-env`

## Uso

Ejecutar el flujo completo: `python flows/collision_flow.py run`

Ver resultados en MLflow: `mlflow ui --backend-store-uri mlruns/`
Abre tu navegador y ve a http://localhost:5000 para ver los experimentos y modelos registrados.