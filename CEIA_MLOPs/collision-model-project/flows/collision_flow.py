import os
os.environ['METAFLOW_DEFAULT_ENVIRONMENT'] = 'local'

from metaflow import FlowSpec, step
import sys
import os

# Añadir src al path para importar los módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_ingestion import ingest_collision_data
from src.data_preprocessing import preprocess_data
from src.train import train_model

class CollisionFlow(FlowSpec):
    
    @step
    def start(self):
        print("Starting collision data processing pipeline")
        self.next(self.ingest_data)
    
    @step
    def ingest_data(self):
        print("Ingesting data from API")
        self.df_raw = ingest_collision_data()
        self.next(self.preprocess_data)
    
    @step
    def preprocess_data(self):
        print("Preprocessing data")
        self.df_processed = preprocess_data(self.df_raw)
        self.next(self.train_model)
    
    @step
    def train_model(self):
        print("Training model with MLflow tracking")
        self.model = train_model()
        self.next(self.end)
    
    @step
    def end(self):
        print("Pipeline completed successfully")

if __name__ == '__main__':
    CollisionFlow()