# 🌍 QA sur les Capitales avec DistilBERT

Ce projet utilise **DistilBERT** pour répondre aux questions sur les capitales du monde en **classification binaire** (normalement `1 = bonne réponse, 0 = mauvaise`).  
L'application est accessible via une **interface web** grâce à Flask.

---

## 🚀 **1. Technologies utilisées**
- **Python** (3.12+)
- **Hugging Face `transformers`** (NLP)
- **DistilBERT** (modèle de classification)
- **Flask** (API web)
- **Pandas** (gestion des données)
- **PyTorch** (deep learning)

---

## 📊 **2. Entraînement du modèle**
### 📌 **a) Installation des dépendances**
Assure-toi d’avoir **Python 3.12+**, puis installe les bibliothèques requises :
```bash
pip install -r requirements.txt