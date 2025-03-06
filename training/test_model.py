import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# 📌 Chargement du modèle et du tokenizer
model_name = "models/distilbert-capital-classification/checkpoint-375"  # Vérifie que ce checkpoint est le bon
tokenizer = DistilBertTokenizerFast.from_pretrained("models/tokenizer_distilbert")  # Assure-toi que ce tokenizer est bien sauvegardé
model = DistilBertForSequenceClassification.from_pretrained(model_name)

# 📌 Liste de questions de test
questions = [
    "Quelle est la capitale de la France ?",
    "Quelle est la capitale du Japon ?",
    "Quelle est la capitale de l'Italie ?",
    "Quelle est la capitale de l'Allemagne ?",
    "Quelle est la capitale du Brésil ?",
    "Quelle est la capitale du Canada ?"
]

print("\n🔎 Test du modèle sur les capitales :\n")
for question in questions:
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    
    # 📌 Affichage des logits pour comprendre les prédictions
    logits = outputs.logits.detach().numpy()
    prediction = torch.argmax(outputs.logits, dim=1).item()
    
    print(f"🔹 Question: {question}")
    print(f"   Logits: {logits}")
    print(f"   ➡ Prédiction: {'Bonne réponse' if prediction == 1 else 'Mauvaise réponse'}\n")