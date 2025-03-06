## Structure du Projet

- **config/**: Configuration globale.
- **data/**: Chargement du dataset et tokenization.
- **models/**: 
  - `transformer_model.py`: Modèle Transformer pour génération (optionnel).
  - `rag_model.py`: Pipeline RAG utilisant DPR/BERT.
- **training/**: Scripts d'entraînement et génération en mode console.
- **utils/**: Fonctions utilitaires.
- **templates/**: Template HTML pour l'interface web.
- **app.py**: Application Flask interactive utilisant RAG.
- **requirements.txt**: Dépendances.
- **README.md**: Documentation.

## Installation

1. Clonez le dépôt.
2. Installez les dépendances :

   ```bash
   pip install -r requirements.txt