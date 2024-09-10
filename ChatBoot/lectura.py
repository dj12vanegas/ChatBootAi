import os
import json
import pandas as pd

def convert_single_to_double_quotes(content):
    """
    Convert single quotes to double quotes in JSON content.
    """
    content = content.replace("'", '"')
    return content

def load_and_combine_json_files(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    # Leer el contenido del archivo
                    content = f.read()
                    
                    # Convertir comillas simples a dobles
                    content = convert_single_to_double_quotes(content)
                    
                    # Manejar múltiples objetos JSON en el archivo
                    json_objects = content.strip().split('\n')
                    for obj in json_objects:
                        try:
                            json_data = json.loads(obj)
                            if isinstance(json_data, dict):
                                examples = json_data.get('rasa_nlu_data', {}).get('common_examples', [])
                            elif isinstance(json_data, list):
                                examples = json_data
                            else:
                                examples = []
                            
                            # Agregar los ejemplos a los datos combinados
                            data.extend(examples)
                        except json.JSONDecodeError as e:
                            print(f"Error al decodificar JSON en el contenido del archivo {filename}: {e}")
                except Exception as e:
                    print(f"Error al procesar el archivo {filename}: {e}")
    return data

# Carpeta con archivos JSON (subcarpeta 'training' dentro de 'dataset')
folder_path = 'dataset/training'

# Cargar y combinar datos
data = load_and_combine_json_files(folder_path)

# Convertir los datos a un DataFrame de pandas
df = pd.DataFrame(data)

# Verifica si los datos se cargaron correctamente
print(df.head())
print(df.info())

# Guardar el DataFrame en un archivo CSV para su posterior uso
df.to_csv('combined_data.csv', index=False, encoding='utf-8')

