import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

# Charger la configuration depuis config.yaml
def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Charger les paramètres depuis config.yaml
model_name = config.get("model_name", "distilbert-base-uncased")
learning_rate = config.get("learning_rate", 2e-5)
num_epochs = config.get("num_epochs", 3)
batch_size = config.get("batch_size", 8)
dataset_path = config.get("dataset_name", "data/dataset_bert.csv")
train_split = config.get("train_split", 0.8)
val_split = config.get("val_split", 0.2)

# Charger le dataset CSV
df = pd.read_csv(dataset_path)

# Séparer en train et validation (80% train, 20% validation)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["question"] + " " + df["capitale"], df["label"], 
    test_size=val_split, random_state=42
)

# Convertir en dataset Hugging Face
train_dataset = Dataset.from_dict({"text": train_texts.tolist(), "label": train_labels.tolist()})
val_dataset = Dataset.from_dict({"text": val_texts.tolist(), "label": val_labels.tolist()})

# Charger le modèle et tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenisation des données
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)

# Enregistrer le tokenizer après l'entraînement
tokenizer.save_pretrained(config.get("tokenizer_save_path", "models/tokenizer_distilbert"))

# Configuration de l'entraînement
training_args = TrainingArguments(
    output_dir=config.get("model_save_path", "models/distilbert-capital-classification"),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    num_train_epochs=num_epochs,
)

# Définir le Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val
)

# Lancer l'entraînement
trainer.train()

# Sauvegarder le modèle fine-tuné
trainer.save_model(config.get("model_save_path", "models/distilbert-capital-classification"))