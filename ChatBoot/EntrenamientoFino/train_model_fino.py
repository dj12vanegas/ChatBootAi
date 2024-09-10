import pandas as pd
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import numpy as np
import os
import torch

# Cargar el modelo y el tokenizador
model = GPT2LMHeadModel.from_pretrained('./results/model')
tokenizer = GPT2Tokenizer.from_pretrained('./results/tokenizer')

# Leer el dataset
def load_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no se encuentra.")
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    print(f"Total de líneas leídas: {len(lines)}")  # Añadido para depuración
    if not lines:
        raise ValueError("El archivo está vacío.")
    conversations = [{'text': line.strip()} for line in lines if line.strip()]
    print(f"Total de líneas no vacías: {len(conversations)}")  # Añadido para depuración
    return pd.DataFrame(conversations)

file_path = './EntrenamientoFino/dataset.txt'
df = load_dataset(file_path)

# Asegurarme de que el DataFrame no esté vacío
if df.empty:
    raise ValueError("El DataFrame está vacío. Verifica el contenido del archivo.")

# Convertir el DataFrame en un dataset de Hugging Face
dataset = Dataset.from_pandas(df)

# Tokenización del dataset
def tokenize_function(examples):
    encodings = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=200)
    encodings['labels'] = encodings['input_ids']
    return encodings

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Asegurarme de que el dataset tokenizado no esté vacío
if len(tokenized_dataset) == 0:
    raise ValueError("El dataset tokenizado está vacío.")

print(f"Longitud del dataset: {len(df)}")
print(f"Longitud del dataset tokenizado: {len(tokenized_dataset)}")
print(f"Entrada tokenizada de muestra: {tokenized_dataset[0]}")

# Calcular perplejidad
def compute_perplexity(eval_preds):
    logits, labels = eval_preds
    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return {"perplexity": np.exp(loss.item())}

# Configurar el entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=False,  # fp16 en falso para el muestreo
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # Puedo usar un dataset separado para evaluación si lo tengo
    compute_metrics=compute_perplexity  # Agregar función de métricas
)

# Entrenar el modelo
trainer.train()

# Guardar el modelo ajustado
model.save_pretrained('./results/model_fino')
tokenizer.save_pretrained('./results/tokenizer_fino')
