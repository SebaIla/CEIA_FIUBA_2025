import requests
import pandas as pd
import time
import os

def ingest_collision_data():
    api_ep_la_collisions = 'https://data.lacity.org/resource/d5tf-ez2w.json'
    LIMIT = 50000
    MAX_RETRIES = 3
    RETRY_DELAY = 10
    
    la_collisions_data = []
    offset = 0
    total_retrieved = 0
    
    print("Starting data retrieval...")
    
    while True:
        retries = 0
        success = False
        
        while retries < MAX_RETRIES and not success:
            try:
                params = {'$limit': LIMIT, '$offset': offset}
                response = requests.get(api_ep_la_collisions, params=params)
                
                if response.status_code == 200:
                    chunk = response.json()
                    la_collisions_data.extend(chunk)
                    retrieved = len(chunk)
                    total_retrieved += retrieved
                    print(f"✅ Partially ingested {retrieved} rows (Total: {total_retrieved})")
                    offset += retrieved
                    success = True
                else:
                    print(f"⚠️ HTTP Error {response.status_code}. Retrying...")
                    retries += 1
                    time.sleep(RETRY_DELAY)
                    
            except Exception as e:
                print(f"⚠️ Exception: {str(e)}. Retrying...")
                retries += 1
                time.sleep(RETRY_DELAY)
        
        if not success:
            print("❌ Failed after maximum retries")
            break
            
        if len(chunk) < LIMIT:
            break
    
    # Save raw data
    df = pd.json_normalize(la_collisions_data)
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/la_collisions_raw.csv', index=False)
    
    return df

if __name__ == "__main__":
    ingest_collision_data()