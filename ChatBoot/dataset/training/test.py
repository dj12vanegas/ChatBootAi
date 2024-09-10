import pandas as pd
import json

def json_to_csv(json_file, csv_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        common_examples = data.get('rasa_nlu_data', {}).get('common_examples', [])
        df = pd.DataFrame(common_examples)
        df.to_csv(csv_file, index=False)
    print(f"El archivo CSV se ha guardado en: {csv_file}")

# Ruta al archivo combinado JSON
combined_file_path = './dataset/training/combined_data.json'
csv_file_path = './dataset/training/combined_data.csv'

json_to_csv(combined_file_path, csv_file_path)
